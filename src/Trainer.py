import torch
import os, time
import logging as log
from Graph import Knowledge_graph
from Environment import Environment
from Baseline import ReactiveBaseline
import numpy as np


class Trainer():
    def __init__(self, option, agent, data_loader):
        self.option = option
        self.agent = agent
        self.data_loader = data_loader
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.option.learning_rate)
        self.positive_reward = torch.tensor(1.)
        self.negative_reward = torch.tensor(0.)
        self.baseline = ReactiveBaseline(option, self.option.Lambda)
        self.decaying_beta = self.option.beta

        if self.option.use_cuda:
            self.agent.cuda()

    def calc_cum_discounted_reward(self, rewards):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.option.max_step_length])

        if self.option.use_cuda:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.option.max_step_length - 1] = rewards
        for t in reversed(range(self.option.max_step_length)):
            running_add = self.option.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar
        return entropy_loss

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):

        loss = torch.stack(all_loss, dim=1)  # [B, T]
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)  # [B, T]
        entropy_loss = self.decaying_beta * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss  # scalar

        return total_loss

    def train(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "w", encoding='UTF-8') as f:
            f.write("Begin train\n")
        if self.option.use_cuda: 
            self.agent.cuda()

        train_graph = Knowledge_graph(self.option, self.data_loader, self.data_loader.get_graph_data())
        train_data = self.data_loader.get_data("train")
        environment = Environment(self.option, train_graph, train_data, "train")

        batch_counter = 0
        current_decay_count = 0

        for start_entities, queries, answers, all_correct in environment.get_next_batch():
            start = time.time()
            if batch_counter > self.option.train_batch:
                break
            else:
                batch_counter += 1

            current_decay_count += 1
            if current_decay_count == self.option.decay_batch:
                self.decaying_beta *= self.option.decay_rate
                current_decay_count = 0

            batch_size = start_entities.shape[0]
            prev_state = [torch.zeros(start_entities.shape[0], self.option.state_embed_size),
                          torch.zeros(start_entities.shape[0], self.option.state_embed_size)]
            # prev_relation = self.agent.get_dummy_start_relation(batch_size)
            prev_relation = queries
            current_entities = start_entities
            queries_cpu = queries.detach().clone()
            if self.option.use_cuda:
                prev_relation = prev_relation.cuda()
                prev_state[0] = prev_state[0].cuda()
                prev_state[1] = prev_state[1].cuda()
                queries = queries.cuda()
                current_entities = current_entities.cuda()

            all_loss = []
            all_logits = []
            all_action_id = []

            sequences = torch.stack((answers, queries_cpu, start_entities), -1)

            for step in range(self.option.max_step_length):
                actions_id = train_graph.get_out(current_entities.detach().clone().cpu(), start_entities, queries_cpu,
                                                 answers, all_correct, step)
                if self.option.use_cuda:
                    actions_id = actions_id.cuda()
                loss, new_state, logits, action_id, next_entities, chosen_relation= \
                    self.agent.step(prev_state, prev_relation, current_entities, actions_id, queries)

                sequences = torch.cat((sequences, chosen_relation.cpu().reshape((sequences.shape[0], -1))), 1)
                sequences = torch.cat((sequences, next_entities.cpu().reshape((sequences.shape[0], -1))), 1)

                all_loss.append(loss)
                all_logits.append(logits)
                all_action_id.append(action_id)
                prev_relation = chosen_relation
                current_entities = next_entities
                prev_state = new_state
                        
            # todo: introduce options for reward selection
            #rewards = self.agent.get_reward(current_entities.cpu(), answers, self.positive_reward, self.negative_reward)
            _, rewards_np, _ = self.agent.get_context_reward(sequences, all_correct)
            rewards = torch.from_numpy(rewards_np * 1.)
            if self.option.use_cuda:
                rewards = rewards.cuda()

            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            reinforce_loss = self.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
            
            if np.isnan(reinforce_loss.detach().cpu().numpy()):
                raise ArithmeticError("Error in computing loss")

            reward_reshape = rewards.detach().cpu().numpy().reshape(self.option.batch_size, self.option.train_times)
            reward_reshape = np.sum(reward_reshape>0.5, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            avg_ep_correct = num_ep_correct / self.option.batch_size

            log.info("{:3.0f} reward: {:2.3f}\tnum ep correct: {:3d}\tavg ep correct: {:3.3f}\tloss: {:3.3f}\t reinforce loss: {:3.3f}"
                     .format(batch_counter, rewards.mean(), num_ep_correct, avg_ep_correct, loss.mean(), reinforce_loss.mean()))
            with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
                f.write("reward: " + str(rewards.mean()) + "\n")
            
            self.baseline.update(torch.mean(cum_discounted_reward))
            self.agent.zero_grad()
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.option.grad_clip_norm, norm_type=2)
            self.optimizer.step()


    def test(self, data='valid', short=False):
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "w", encoding='UTF-8') as f:
            f.write("Begin test on {} data\n".format(data))
        with torch.no_grad():
            if self.option.use_cuda:
                self.agent.cpu()
                self.agent.path_scoring_model.cuda()
                torch.cuda.empty_cache() 
                #self.option.use_cuda = False

            test_graph = Knowledge_graph(self.option, self.data_loader, self.data_loader.get_graph_data())
            test_data = self.data_loader.get_data(data)
            test_graph.update_all_correct(self.data_loader.get_data('valid'))
            test_graph.update_all_correct(self.data_loader.get_data('test'))

            environment = Environment(self.option, test_graph, test_data, 'test')

            total_examples = len(test_data) if not short else (self.option.test_batch_size * short)

            # introduce left / right evaluation
            metrics = np.zeros((6,3))
            num_right = len(self.data_loader.data[data])
            first_left_batch, split = num_right // self.option.test_batch_size, num_right % self.option.test_batch_size
            
            # todo: add sequences for the standard Minerva
            # _variable + all correct: with rollouts; variable: original data
            for i, (_start_entities, _queries, _answers, start_entities, queries, answers, all_correct)\
                    in enumerate(environment.get_next_batch(short)):

                batch_size = len(start_entities)
                sequences = torch.stack((_answers, _queries, _start_entities), -1).reshape(batch_size, -1, 3)
                prev_state = [torch.zeros(start_entities.shape[0], self.option.state_embed_size),
                              torch.zeros(start_entities.shape[0], self.option.state_embed_size)]
                # prev_relation = self.agent.get_dummy_start_relation(start_entities.shape[0])
                prev_relation = queries
                current_entities = start_entities
                log_current_prob = torch.zeros(start_entities.shape[0])
                
                for step in range(self.option.max_step_length):
                    if step == 0:
                        actions_id = test_graph.get_out(current_entities, start_entities, queries, answers, all_correct,
                                                        step)
                        chosen_state, chosen_relation, chosen_entities, log_current_prob, sequences = \
                            self.agent.test_step(prev_state, prev_relation, current_entities, actions_id,
                                                 log_current_prob, queries, batch_size, sequences,
                                                 step == self.option.max_step_length - 1)

                    else:
                        actions_id = test_graph.get_out(current_entities, _start_entities, _queries, _answers,
                                                        all_correct, step)
                        chosen_state, chosen_relation, chosen_entities, log_current_prob, sequences = \
                            self.agent.test_step(prev_state, prev_relation, current_entities, actions_id,
                                                 log_current_prob, _queries, batch_size, sequences,
                                                 step == self.option.max_step_length - 1)

                    prev_relation = chosen_relation
                    current_entities = chosen_entities
                    prev_state = chosen_state

                # B x TIMES
                # todo: flexible reward
                #rewards = self.agent.get_reward(current_entities, _answers,  self.positive_reward, self.negative_reward)
                #rewards = rewards.reshape(-1, self.option.test_times).detach().cpu().numpy()
                
                top_k_rewards_np, rewards_np, ranks_np = self.agent.get_context_reward(
                    sequences.squeeze(1), all_correct[::self.option.test_times], test=True)
                
                #assert sequences.shape[0] == batch_size == len(all_correct[::self.option.test_times])
                # todo: unify output shape of beam search
                # if self.option.use_cuda:
                #     current_entities = current_entities.cpu()
                # current_entities_np = current_entities.numpy()
                # current_entities_np = current_entities_np.reshape(-1, self.option.test_times)

                # for line_id in range(rewards.shape[0]):
                #     seen = set()
                #     pos = 0
                #     find_ans = False
                #     for loc_id in range(rewards.shape[1]):
                #         if rewards[line_id][loc_id] == self.positive_reward:
                #             find_ans = True
                #             break
                #         if current_entities_np[line_id][loc_id] not in seen:
                #             seen.add(current_entities_np[line_id][loc_id])
                #             pos += 1
                # if find_ans:
                #     if pos < 20:
                #         final_reward_20 += 1
                #         if pos < 10:
                #             final_reward_10 += 1
                #             if pos < 5:
                #                 final_reward_5 += 1
                #                 if pos < 3:
                #                     final_reward_3 += 1
                #                     if pos < 1:
                #                         final_reward_1 += 1
                #     r_rank += 1.0 / (pos + 1)
                # else:
                #     r_rank += 0  # an appropriate last rank = 1.0 / self.data_loader.num_entity, but no big difference

                metrics[:, 2] += self.get_metrics(ranks_np)

                if i < first_left_batch:
                    metrics[:, 1] += self.get_metrics(ranks_np)
                elif i == first_left_batch:  ## batches with inverse relations began
                    metrics[:, 1] += self.get_metrics(ranks_np[:split])
                    metrics[:, 0] += self.get_metrics(ranks_np[split:])
                else:
                    metrics[:, 0] += self.get_metrics(ranks_np)

            assert (metrics[:5, 2] == (metrics[:5, 0] + metrics[:5, 1])).all()

            # total counts
            metrics[:, 2:] /= total_examples
            metrics[:, :2] /= num_right
            
            log.info(("all_final_reward_1", metrics[4]))
            log.info(("all_final_reward_3", metrics[3]))
            log.info(("all_final_reward_5", metrics[2]))
            log.info(("all_final_reward_10", metrics[1]))
            log.info(("all_final_reward_20", metrics[0]))
            log.info(("all_r_rank", metrics[5]))
            
            with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
                f.write("all_final_reward_1: " + str(metrics[4]) + "\n")
                f.write("all_final_reward_3: " + str(metrics[3]) + "\n")
                f.write("all_final_reward_5: " + str(metrics[2]) + "\n")
                f.write("all_final_reward_10: " + str(metrics[1]) + "\n")
                f.write("all_final_reward_20: " + str(metrics[0]) + "\n")
                f.write("all_r_rank: " + str(metrics[5]) + "\n")

    def get_metrics(self, ranks_np):
        metrics = np.zeros(6)
        for pos in ranks_np:
            if pos < 20:
                metrics[0] += 1
                if pos < 10:
                    metrics[1] += 1
                    if pos < 5:
                        metrics[2] += 1
                        if pos < 3:
                            metrics[3] += 1
                            if pos < 1:
                                metrics[4] += 1

            metrics[5] += 1.0 / (pos + 1)
        return metrics

    def decode_and_save_paths(self, paths, data):
        pass
        #with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
        #    pass


    def save_model(self):
        path = os.path.join(self.option.this_expsdir, "model.pkt")
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        self.agent.cpu()
        torch.save(self.agent.state_dict(), path)

    def load_model(self):
        if self.option.load_model:
            dir_path = os.path.join(self.option.exps_dir, self.option.load_model)
        else:
            dir_path = self.option.this_expsdir
        path = os.path.join(dir_path, "model.pkt")
        self.agent.load_state_dict(torch.load(path))
        
