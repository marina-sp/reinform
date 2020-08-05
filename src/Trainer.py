import torch
import os, time
import logging as log
from Graph import Knowledge_graph
from Environment import Environment
from Baseline import ReactiveBaseline, RandomBaseline
import numpy as np


class Trainer():
    def __init__(self, option, agent, data_loader):
        self.option = option
        self.agent = agent
        self.data_loader = data_loader
        # train bert with smaller rate
        self.optimizer = torch.optim.Adam([
                {'params': self.agent.non_bert_parameters},
                {'params': self.agent.path_scoring_model.parameters(), 'lr': self.option.bert_lr}
            ],
            lr=self.option.learning_rate)
        self.positive_reward = torch.tensor(1.)
        self.negative_reward = torch.tensor(0.)

        if option.baseline == "random" and option.reward == "context":
            self.baseline = RandomBaseline(option, agent)
        elif option.baseline == "react":
            self.baseline = ReactiveBaseline(option, self.option.Lambda)
        else:
            raise RuntimeError("random baseline incompatible with answer search")
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

    def calc_reinforce_loss(self, all_loss, all_logits, final_reward):

        loss = torch.stack(all_loss, dim=1)  # [B, T]
        #base_value = self.baseline.get_baseline_value()
        #final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)
        #final_reward /= final_reward.max()

        loss = torch.mul(loss, final_reward)  # [B, T]
        entropy_loss = self.decaying_beta * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss  # scalar

        return total_loss, final_reward


    def train(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "w", encoding='UTF-8') as f:
            f.write("Begin train\n")
        if self.option.use_cuda: 
            self.agent.cuda()

        train_graph = Knowledge_graph(self.option, self.data_loader, self.data_loader.get_graph_data())
        train_data = self.data_loader.get_data("valid" if self.option.reward == "context" else "train")
        environment = Environment(self.option, train_graph, train_data, "train")

        batch_counter = 0
        current_decay_count = 0

        print_loss = 0.
        print_rewards = 0.
        print_act_loss = 0.

        for start_entities, queries, answers, all_correct in environment.get_next_batch():
            start = time.time()
            if batch_counter >= self.option.train_batch:
                break
            else:
                batch_counter += 1

            current_decay_count += 1
            if current_decay_count == self.option.decay_batch:
                self.decaying_beta *= self.option.decay_rate
                current_decay_count = 0

            batch_size = start_entities.shape[0]
            self.agent.zero_state(batch_size)

            if self.option.reward == "answer":
                prev_relation = self.agent.get_dummy_start_relation(batch_size)
                sequences = torch.empty((batch_size, 0), dtype=queries.dtype)
            else:
                prev_relation = queries
                sequences = torch.stack((answers, queries, start_entities), -1)

            current_entities = start_entities
            queries_cpu = queries.detach().clone()
            if self.option.use_cuda:
                prev_relation = prev_relation.cuda()
                queries = queries.cuda()
                current_entities = current_entities.cuda()

            all_loss = []
            all_logits = []
            all_action_id = []

            for step in range(self.option.max_step_length):
                actions_id = train_graph.get_out(current_entities.detach().clone().cpu(), start_entities, queries_cpu,
                                                 answers, all_correct, step)
                if self.option.use_cuda:
                    actions_id = actions_id.cuda()
                loss, logits, action_id, next_entities, chosen_relation= \
                    self.agent.step(prev_relation, current_entities, actions_id, queries, sequences,
                                    self.option.mode == "random")

                sequences = torch.cat((sequences, chosen_relation.cpu().reshape((sequences.shape[0], -1))), 1)
                sequences = torch.cat((sequences, next_entities.cpu().reshape((sequences.shape[0], -1))), 1)

                all_loss.append(loss)
                all_logits.append(logits)
                all_action_id.append(action_id)
                prev_relation = chosen_relation
                current_entities = next_entities

            if self.option.reward == "answer":
                rewards = self.agent.get_reward(current_entities.cpu(), answers, self.positive_reward, self.negative_reward)
            elif self.option.reward == "context":
                bert_loss, rewards, _ = self.agent.get_context_reward(sequences, all_correct)
                if self.option.use_cuda:
                    rewards = rewards.cuda()

            # apply baseline deduction

            # cum_discounted_reward = self.calc_cum_discounted_reward(rewards).detach()
            base_value = self.baseline.get_baseline_value(
                batch=(start_entities, start_entities, queries, answers, all_correct),
                graph=train_graph)
            if base_value.shape[0] != 1:
                base_value = self.calc_cum_discounted_reward(base_value)
            # final_reward = cum_discounted_reward - base_value
            
            base_rewards = rewards - base_value
            cum_discounted_reward = self.calc_cum_discounted_reward(base_rewards).detach()
            reinforce_loss, norm_reward = self.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
            #bert_loss = -(rewards.log() * base_rewards.clamp_min(0).detach()).mean()

            reinforce_loss += self.option.bert_rate * bert_loss

            if np.isnan(reinforce_loss.detach().cpu().numpy()):
                raise ArithmeticError("Error in computing loss")

            reward_reshape = rewards.detach().cpu().numpy().reshape(self.option.batch_size, self.option.train_times)
            reward_reshape = np.sum(reward_reshape>0.5, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            avg_ep_correct = num_ep_correct / self.option.batch_size

            print_loss = 0.9 * print_loss + 0.1 * reinforce_loss.item()
            print_rewards = 0.9 * print_rewards + 0.1 * rewards.mean()
            print_act_loss = 0.9 * print_act_loss + 0.1 * torch.stack(all_loss).mean()
            log.info("{:3.0f} sliding reward: {:2.3f}\t red reward: {:2.3f}\tnum ep correct: {:3d}\tavg ep correct: {:3.3f}\t sliding act loss: {:3.3f}\t sliding reinforce loss: {:3.3f}"
                     .format(batch_counter, print_rewards, norm_reward.mean(), num_ep_correct, avg_ep_correct,
                              print_act_loss, print_loss))
            with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
                f.write("reward: " + str(rewards.mean()) + "\n")
            
            self.baseline.update(torch.mean(base_rewards))
            self.optimizer.zero_grad()
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.option.grad_clip_norm, norm_type=2)
            self.optimizer.step()


    def test(self, data='valid', short=False):
        with open(os.path.join(self.option.this_expsdir, f"{data}_log.txt"), "w", encoding='UTF-8') as f:
            f.write(f"Begin test on {data} data\n")
        with open(os.path.join(self.option.this_expsdir, f"{data}_paths.txt"), "w", encoding='UTF-8') as f:
            f.write("")
        with torch.no_grad():
            self.agent.eval()
            if self.option.use_cuda:
                self.agent.cpu()
                if "context" in [self.option.reward, self.option.metric]:
                    self.agent.path_scoring_model.cuda()
                torch.cuda.empty_cache()
                self.option.use_cuda = False

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
            print(len(test_data), num_right, self.option.test_batch_size, first_left_batch, split)

            # _variable + all correct: with rollouts; variable: original data
            for i, (_start_entities, _queries, _answers, start_entities, queries, answers, all_correct)\
                    in enumerate(environment.get_next_batch(short)):

                batch_size = len(start_entities)
                self.agent.zero_state(batch_size)
                if self.option.reward == "answer":
                    prev_relation = self.agent.get_dummy_start_relation(batch_size)
                    sequences = start_entities.reshape(-1, 1)
                else:
                    prev_relation = queries
                    sequences = torch.stack((answers, queries, start_entities), -1)#.reshape(batch_size, -1, 3)

                current_entities = start_entities
                log_current_prob = torch.zeros(start_entities.shape[0])
                
                for step in range(self.option.max_step_length):
                    if step == 0:
                        actions_id = test_graph.get_out(current_entities, start_entities, queries, answers, all_correct,
                                                        step)
                        chosen_relation, chosen_entities, log_current_prob, sequences = self.agent.test_step(
                            prev_relation, current_entities, actions_id,
                            log_current_prob, queries, batch_size, sequences,
                            step,
                            self.option.mode == "random")

                    else:
                        actions_id = test_graph.get_out(current_entities, _start_entities, _queries, _answers,
                                                        all_correct, step)
                        chosen_relation, chosen_entities, log_current_prob, sequences = self.agent.test_step(
                            prev_relation, current_entities, actions_id,
                            log_current_prob, _queries, batch_size, sequences,
                            step,
                            self.option.mode == "random")

                    prev_relation = chosen_relation
                    current_entities = chosen_entities


                if (self.option.reward == "context") and (self.option.metric == "context"):
                    sequences = sequences #.squeeze(1)
                    triples = torch.stack((answers, queries, start_entities), dim=-1)
                elif (self.option.reward == "answer") and (self.option.metric == "context"):
                    # post-process sequences from Minerva for context evaluation
                    # - save top 1
                    sequences = sequences[::self.option.test_times, :]
                    # - add reversed query to the path
                    # t=mask rel_inv h=start_entities -- path
                    inv_queries = torch.tensor([
                        self.data_loader.kg.rel2inv[rel.item()] for rel in queries
                    ]).to(queries.device)
                    sequences = torch.cat((answers.view(-1,1), inv_queries.view(-1,1), sequences), -1)
                    triples = torch.stack((start_entities, queries, answers), dim=1)
                elif (self.option.reward == "answer") and (self.option.metric == "answer"):
                    # sequences can be printed as is - but only the top 1
                    sequences = sequences[::self.option.test_times, :]
                    triples = torch.stack((start_entities, queries, answers), dim=-1)
                    pass

                if self.option.metric == "context":
                    # - pad NO_OP
                    _, _, ranks_np = self.agent.get_context_reward(
                        sequences, all_correct[::self.option.test_times], test=True)

                elif self.option.metric == "answer":
                    rewards = self.agent.get_reward(current_entities, _answers,  self.positive_reward, self.negative_reward)
                    rewards = rewards.reshape(-1, self.option.test_times).detach().cpu().numpy()

                    current_entities_np = current_entities.numpy()
                    current_entities_np = current_entities_np.reshape(-1, self.option.test_times)

                    ranks = []
                    for line_id in range(rewards.shape[0]):
                        seen = set()
                        pos = 0
                        find_ans = False
                        for loc_id in range(rewards.shape[1]):
                            if rewards[line_id][loc_id] == self.positive_reward:
                                find_ans = True
                                ranks.append(pos)
                                break
                            if current_entities_np[line_id][loc_id] not in seen:
                                seen.add(current_entities_np[line_id][loc_id])
                                pos += 1
                        if not find_ans:
                            # an appropriate last rank = self.data_loader.num_entity, if not found but no big difference
                            ranks.append(self.option.num_entity)
                    ranks_np = np.array(ranks)

                self.decode_and_save_paths(triples, sequences, ranks_np, data)

                metrics[:, 2] += self.get_metrics(ranks_np)

                if i < first_left_batch:
                    metrics[:, 0] += self.get_metrics(ranks_np)
                elif i == first_left_batch:  ## batches with inverse relations began
                    metrics[:, 0] += self.get_metrics(ranks_np[:split])
                    metrics[:, 1] += self.get_metrics(ranks_np[split:])
                else:
                    metrics[:, 1] += self.get_metrics(ranks_np)

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

            with open(os.path.join(self.option.this_expsdir, f"{data}_log.txt"), "a+", encoding='UTF-8') as f:
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

    def decode_and_save_paths(self, queries, sequences, ranks, data):
        str_qs   = [" ".join([self.data_loader.num2entity[h],
                              self.data_loader.num2relation[r],
                              self.data_loader.num2entity[t]])
                    for h,r,t in queries.numpy()]
        str_ents = [[self.data_loader.num2entity[idx] for idx in seq] for seq in sequences[:, ::2].numpy()]
        str_rels = [[self.data_loader.num2relation[idx] for idx in seq] for seq in sequences[:, 1::2].numpy()]

        with open(os.path.join(self.option.this_expsdir, f"{data}_paths.txt"), "a+", encoding='UTF-8') as f:
            for qid, q in enumerate(str_qs):
                out = [str_ents[qid][0]]
                for step in range(len(str_rels[0])):
                    out.append(str_rels[qid][step])
                    out.append(str_ents[qid][step+1])
                path = " ".join(out)
                rank = "{:3d}".format(ranks[qid]+1)
                f.write("\t".join((q, path, rank)) + "\n")

    def save_model(self):
        path = os.path.join(self.option.this_expsdir, "model.pkt")
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        self.agent.cpu()
        torch.save(self.agent.my_state_dict(), path)

    def load_model(self):
        if self.option.mode == "random":
            return
        if self.option.load_model:
            dir_path = os.path.join(self.option.exps_dir, self.option.load_model)
        else:
            dir_path = self.option.this_expsdir
        path = os.path.join(dir_path, "model.pkt")
        state_dict = {k:v for k,v in torch.load(path).items()}  # if not k.startswith('path')}

        log.info(f"load model from: {dir_path}\n")
        log.info("loaded parameters: {}\n".format(state_dict.keys()))

        self.agent.load_state_dict(state_dict, strict=False)
