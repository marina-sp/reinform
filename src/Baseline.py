import torch


class ReactiveBaseline():
    def __init__(self, option, update_rate):
        self.option = option
        self.update_rate = update_rate
        self.value = torch.zeros(1)
        if self.option.use_cuda:
            self.value = self.value.cuda()

    def get_baseline_value(self, **params):
        return self.value

    def update(self, target):
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target).detach()

class RandomBaseline():
    def __init__(self, option, agent):
        self.option = option
        self.agent = agent

    def get_baseline_value(self, batch, graph):
        current_entities, start_entities, queries, answers, all_correct = batch

        batch_size = start_entities.shape[0]
        self.agent.zero_state(batch_size)
        # prev_relation = self.agent.get_dummy_start_relation(batch_size)
        prev_relation = queries
        current_entities = start_entities
        queries_cpu = queries.detach().clone().cpu()
        if self.option.use_cuda:
            prev_relation = prev_relation.cuda()
            queries = queries.cuda()
            current_entities = current_entities.cuda()

        sequences = torch.stack((answers, queries_cpu, start_entities), -1)
        for step in range(self.option.max_step_length):
            actions_id = graph.get_out(current_entities.detach().clone().cpu(), start_entities, queries_cpu,
                                       answers, all_correct, step)
            if self.option.use_cuda:
                actions_id = actions_id.cuda()
            loss, logits, action_id, next_entities, chosen_relation = \
                self.agent.step(prev_relation, current_entities, actions_id, queries, sequences, True)

            sequences = torch.cat((sequences, chosen_relation.cpu().reshape((sequences.shape[0], -1))), 1)
            sequences = torch.cat((sequences, next_entities.cpu().reshape((sequences.shape[0], -1))), 1)

            prev_relation = chosen_relation
            current_entities = next_entities

        _, rewards_np, _ = self.agent.get_context_reward(sequences, all_correct)
        rewards = torch.from_numpy(rewards_np * 1.)
        if self.option.use_cuda:
            rewards = rewards.cuda()

        return rewards

    def update(self, target):
        pass
