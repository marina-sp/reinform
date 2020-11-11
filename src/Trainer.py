import torch
import os, time
import logging as log
from Graph import Knowledge_graph
from Environment import Environment
from Baseline import ReactiveBaseline, RandomBaseline
from sequence import State
import numpy as np
from copy import deepcopy

class Trainer():
    def __init__(self, option, agent, data_loader):
        self.option = option
        self.agent = agent
        self.data_loader = data_loader
        self.graph = Knowledge_graph.get_train_graph(self.option, self.data_loader)
        self.test_graph = Knowledge_graph.get_test_graph(self.option, self.data_loader)
        assert(self.graph != self.test_graph)
        self.train_data = 'train'
        self.valid_data = 'valid'
        self.valid_idx = np.random.RandomState(self.option.random_seed)\
            .randint(0,
                     len(self.data_loader.get_data(self.valid_data, include_inverse=True)),
                     size=len(self.data_loader.get_data(self.valid_data, include_inverse=True)))

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.option.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, min_lr=1e-1, verbose=True, patience=100, mode="max")

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
            self.device = torch.device('cuda')
            self.agent.to(self.device)
        else:
            self.device = torch.device('cpu')

    def calc_cum_discounted_reward(self, rewards):
        # normalization by episode
        reward_by_episode = rewards.reshape(-1 , self.option.train_times)
        reward_mean = torch.mean(reward_by_episode, 1, keepdim=True)
        reward_std = torch.std(reward_by_episode, 1, keepdim=True) + 1e-6
        #print(reward_by_episode.shape, reward_mean.shape, reward_std.shape)
        rewards = torch.div(reward_by_episode - reward_mean, reward_std).flatten() #.clamp_min_(0).flatten()
        print("mean interval of rewards by episode: ", torch.mean(reward_by_episode.max(dim=-1)[0] - reward_by_episode.min(dim=-1)[0]))
        #print(rewards.mean(-1))
        #base_rewards = rewards.reshape(-1)
        #print(rewards.mean())
        #final_reward /= final_reward.max()

        # baseline reduction
        #rewards = base_rewards - self.baseline.get_baseline_value()
        #self.baseline.update(base_rewards.mean())
        
        #neutralize middle rewards
        #maxk = 1 + round(.75 * (rewards.numel() - 1))
        #mink = 1 + round(.25 * (rewards.numel() - 1))
        #maxv = torch.kthvalue(rewards, maxk).values.item()
        #minv = torch.kthvalue(rewards, mink).values.item()
        #rewards[(rewards < maxk) & (rewards > mink)] = 0    

        # discounting
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.option.max_step_length])

        if self.option.use_cuda:
            running_add = running_add.to(self.device)
            cum_disc_reward = cum_disc_reward.to(self.device)

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
        #final_reward = reward - base_value
        #self.baseline.update(reward.mean())


        #reward_mean = torch.mean(final_reward)
        #reward_std = torch.std(final_reward) + 1e-6
        #print(reward_by_episode.shape, reward_mean.shape, reward_std.shape)
        #final_reward = torch.div(final_reward - reward_mean, reward_std)
        #print(final_reward.mean(-1))
        #final_reward = final_reward.reshape(-1, self.option.max_step_length)
        #print(final_reward.mean())
        #final_reward /= final_reward.max()

        loss = torch.mul(loss, final_reward)  # [B, T]
        entropy_loss = self.decaying_beta * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss  # scalar

        return total_loss, final_reward


    def train(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "w", encoding='UTF-8') as f:
            f.write("Begin train\n")
        if self.option.use_cuda: 
            self.agent.to(self.device)

        train_data = self.data_loader.get_data(self.train_data, include_inverse=self.option.use_inverse)
        environment = Environment(self.option, self.graph, train_data, "train")

        batch_counter = 0
        current_decay_count = 0

        print_loss = 0.
        print_rewards = 0.
        print_act_loss = 0.

        best_metric = 0.0

        for i, (start_entities, queries, answers, all_correct) in enumerate(environment.get_next_batch()):
            self.agent.train()
            self.agent.test_mode = False

            start = time.time()
            if batch_counter >= self.option.train_batch:
                break
            else:
                batch_counter += 1

            current_decay_count += 1
            if current_decay_count == self.option.decay_batch:
                self.decaying_beta *= self.option.decay_rate
                self.option.epsilon *= self.option.decay_rate
                current_decay_count = 0

            batch_size = start_entities.shape[0]
            self.agent.zero_state(batch_size)

            if self.option.reward == "answer":
                state = State(self.option,
                              self.data_loader.vocab,
                              n_seq=batch_size,
                              query_ent=start_entities,
                              query_rel=queries,
                              answers=answers,
                              graph=self.graph,
                              all_correct=all_correct)
            else:
                state = State(self.option,
                              self.data_loader.vocab,
                              data=answers,
                              query_ent=start_entities,
                              query_rel=queries,
                              answers=answers,
                              graph=self.graph,
                              all_correct=all_correct)

            queries_cpu = state.get_query_rel().detach().clone()

            all_loss = []
            all_logits = []
            all_action_id = []

            for step in range(self.option.max_step_length):

                loss, logits, action_id, chosen_action = self.agent.step(state, step)

                # prev rel and curr ents are updated internally
                state.add_steps(chosen_action)

                all_loss.append(loss)
                all_logits.append(logits)
                all_action_id.append(action_id)

            if self.option.reward == "answer":
                rewards = self.agent.get_reward(
                    state.get_current_ent(hide=True), answers,  # todo: need to hide?
                    self.positive_reward, self.negative_reward)
                bert_loss = 0 
            elif self.option.reward == "context":
                bert_loss, rewards, _ = self.agent.get_context_reward(state.get_context_path(), all_correct)
                if self.option.use_cuda:
                    rewards = rewards.to(self.device)

            # apply baseline deduction

            # cum_discounted_reward = self.calc_cum_discounted_reward(rewards).detach()
            #base_value = self.baseline.get_baseline_value(
            #    batch=(start_entities, start_entities, queries, answers, all_correct),
            #    graph=self.graph)
            #if base_value.shape[0] != 1:
            #    base_value = self.calc_cum_discounted_reward(base_value)
            # final_reward = cum_discounted_reward - base_value
            
            #base_rewards = rewards #- base_value
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards).detach()            
            reinforce_loss, norm_reward = self.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
            #bert_loss = -(rewards.log() * base_rewards.clamp_min(0).detach()).mean()
            
            #reinforce_loss += self.option.bert_rate * bert_loss# * base_rewards.detach().mean().clamp_min(0)

            if np.isnan(reinforce_loss.detach().cpu().numpy()):
                raise ArithmeticError("Error in computing loss")

            if i % self.option.eval_batch == 0:
                valid_mrr = self.test(self.valid_data, 20, verbose=False)
                if valid_mrr > best_metric:
                    best_metric = valid_mrr
                    self.save_model('best')
                    print('saved new best model')
                    #if self.option.use_cuda:
                    #    self.agent.to(self.device)

            reward_reshape = rewards.detach().cpu().numpy().reshape(self.option.batch_size, self.option.train_times)
            reward_reshape = np.sum(reward_reshape>0.5, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            avg_ep_correct = num_ep_correct / self.option.batch_size

            print_loss = 0.9 * print_loss + 0.1 * reinforce_loss.item()
            print_rewards = 0.9 * print_rewards + 0.1 * rewards.mean()
            print_act_loss = 0.9 * print_act_loss + 0.1 * torch.stack(all_loss).mean()
            log.info("{:3.0f} sliding reward: {:2.3f}\t red reward: {:2.3f}\t valid mrr: {:2.3f}\t sliding act loss: {:3.3f}\t sliding reinforce loss: {:3.3f}"
                     .format(batch_counter, print_rewards, cum_discounted_reward.mean(), valid_mrr,
                              print_act_loss, print_loss))
            with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
                f.write(f"reward: {rewards.mean().item()} \t action loss: {torch.stack(all_loss).mean().item()} \t reinforce_loss: {reinforce_loss.item()}\n")

            #self.baseline.update(torch.mean(cum_discounted_reward))
            self.optimizer.zero_grad()
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.option.grad_clip_norm, norm_type=2)
            self.optimizer.step()
            self.scheduler.step(valid_mrr)


    def test(self, data='valid', short=False, verbose=True):
        with open(os.path.join(self.option.this_expsdir, f"{data}_log.txt"), "w", encoding='UTF-8') as f:
            f.write(f"Begin test on {data} data\n")
        with open(os.path.join(self.option.this_expsdir, f"{data}_paths.txt"), "w", encoding='UTF-8') as f:
            f.write("")
        with torch.no_grad():
            self.agent.eval()
            #self.agent.path_scoring_model.path_scoring_model.eval()
            self.agent.test_mode = True

            # todo: clean-up
            if not short and self.option.use_cuda:
                #self.agent.cpu()
                #if "context" in [self.option.reward, self.option.metric]:
                #    self.agent.path_scoring_model.to(self.device)
                torch.cuda.empty_cache()
                #self.option.use_cuda = False

            # for test do not include inverted triples, as an emerging entity can not be predicted
            self.test_env = Environment(self.option, self.test_graph, self.data_loader.get_data(data, True),
                                        'test', self.valid_idx if short else None)
            total_examples = len(self.data_loader.get_data(data)) if not short else (self.option.test_batch_size * short)

            # todo: make clean test on head/test predictions
            # introduce left / right evaluation
            metrics = np.zeros((6,3))
            num_right = len(self.data_loader.data[data])
            first_left_batch, split = num_right // self.option.test_batch_size, num_right % self.option.test_batch_size
            #print(len(test_data), num_right, self.option.test_batch_size, first_left_batch, split)

            # _variable + all correct: with rollouts; variable: original data
            for i, (start_entities, queries, answers, all_correct)\
                    in enumerate(self.test_env.get_next_batch(short)):

                batch_size = len(start_entities)
                self.agent.zero_state(batch_size)
                if self.option.reward == "answer":
                    state = State(self.option,
                                  self.data_loader.vocab,
                                  n_seq=batch_size,
                                  query_ent=start_entities,
                                  query_rel=queries,
                                  answers=answers,
                                  graph=self.test_graph,
                                  all_correct=all_correct)
                else:
                    state = State(self.option,
                                  self.data_loader.vocab,
                                  data=answers,
                                  query_ent=start_entities,
                                  query_rel=queries,
                                  answers=answers,
                                  graph=self.test_graph,
                                  all_correct=all_correct)

                log_current_prob = torch.zeros(start_entities.shape[0]).to(self.device)
                
                for step in range(self.option.max_step_length):

                    if step == 0:
                        # no rollouts in the first iteration
                        chosen_action, log_current_prob, state = self.agent.test_step(
                            log_current_prob, batch_size, state,
                            step)

                    else:
                        chosen_action, log_current_prob, state = self.agent.test_step(
                            log_current_prob, batch_size, state,
                            step)

                    # append new elements to the sequences
                    state.add_steps(chosen_action)

                # adjust sequence format to the evaluation method
                sequences, triples = state.get_output_path(self.option.metric, self.option.test_times)

                if self.option.metric == "context":
                    # - pad NO_OP
                    _, _, ranks_np = self.agent.get_context_reward(sequences, all_correct)

                elif self.option.metric == "answer":
                    current_entities = state.get_current_ent()
                    answers = state.get_answer(do_rollout=True)
                    rewards = self.agent.get_reward(current_entities, answers,  self.positive_reward, self.negative_reward)
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
                
                if verbose:
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

            if verbose:
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

            return metrics[5][2]

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
        str_qs   = [" ".join([self.data_loader.vocab.num2item[h],
                              self.data_loader.vocab.num2item[r],
                              self.data_loader.vocab.num2item[t]])
                    for h,r,t in queries.numpy()]
        str_ents = [[self.data_loader.vocab.num2item[idx] for idx in seq] for seq in sequences[:, ::2].numpy()]
        str_rels = [[self.data_loader.vocab.num2item[idx] for idx in seq] for seq in sequences[:, 1::2].numpy()]

        with open(os.path.join(self.option.this_expsdir, f"{data}_paths.txt"), "a+", encoding='UTF-8') as f:
            for qid, q in enumerate(str_qs):
                out = [str_ents[qid][0]]
                for step in range(len(str_rels[0])):
                    out.append(str_rels[qid][step])
                    out.append(str_ents[qid][step+1])
                path = " ".join(out)
                rank = "{:3d}".format(ranks[qid]+1)
                f.write("\t".join((q, path, rank)) + "\n")

    def save_model(self, name='best'):
        path = os.path.join(self.option.this_expsdir, f"{name}_model.pkt")
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        torch.save(self.agent.my_state_dict(), path)

    def load_model(self, name='best', exp_name=''):
        if self.option.mode == "random":
            return
        if exp_name != '':
            dir_path = os.path.join(self.option.exps_dir, exp_name)
        else:
            dir_path = self.option.this_expsdir
        path = os.path.join(dir_path, f"{name}_model.pkt")
        state_dict = {k:v for k,v in torch.load(path).items()}  # if not k.startswith('path')}

        log.info(f"load model from: {dir_path}\n")
        log.info("loaded {} parameters\n".format(list(state_dict.keys())))

        self.agent.load_state_dict(state_dict, strict=False)
