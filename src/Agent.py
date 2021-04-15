import torch.nn as nn
import numpy as np
import torch
import logging as log
from torch.distributions.categorical import Categorical
from collections import defaultdict
import copy
from transformers import BertForMaskedLM, BertConfig
from coke import CoKEWrapper

class Policy_step(nn.Module):
    def __init__(self, option):
        super(Policy_step, self).__init__()
        self.option = option
        self.lstm_cell = torch.nn.LSTMCell(input_size=self.option.action_embed_size,
                          hidden_size=self.option.state_embed_size)
    '''
    - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
    - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
    - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    '''
    def forward(self, prev_action, prev_state):
        output, new_state = self.lstm_cell(prev_action, prev_state)
        return output, (output, new_state)

class Policy_mlp(nn.Module):
    def __init__(self, option):
        super(Policy_mlp, self).__init__()
        self.option = option
        self.hidden_size = option.mlp_hidden_size
        self.mlp_l1 = nn.Linear(self.option.state_embed_size + self.option.action_embed_size,
                                self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(self.hidden_size, self.option.action_embed_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden)).unsqueeze(1)
        return output

class Bert_policy(nn.Module):
    def __init__(self, data_loader, path_model, option):
        super().__init__()
        self.option = option
        self.to_state = nn.Linear(256, self.option.state_embed_size)

    def forward(self, seq_embs, *params):
        if not self.option.use_cuda:
            seq_embs = seq_embs.cpu()
        # print(embs.shape, input.shape)
        #embs = embs.detach()
        if self.option.bert_state_mode == "avg_token":
            new_state = torch.tanh(self.to_state(seq_embs[:, 1:-1, :].mean(1)))
        elif self.option.bert_state_mode == "avg_all":
            new_state = torch.tanh(self.to_state(seq_embs.mean(1)))
        elif self.option.bert_state_mode == "sep":
            new_state = torch.tanh(self.to_state(seq_embs[:, -1, :]))
        elif self.option.bert_state_mode == "mask":
            new_state = torch.tanh(self.to_state(seq_embs[:, 1, :]))
        return new_state, torch.tensor(0.0)


class Agent(nn.Module):
    def __init__(self, option, data_loader, graph=None):
        super(Agent, self).__init__()
        self.option = option
        self.data_loader = data_loader
        self.dropout = torch.nn.Dropout(self.option.droprate)
        self.token_droprate = self.option.token_droprate
        self.test_mode = False

        # use joint id space from Transformer
        self.item_embedding = nn.Embedding(self.option.num_relation + self.option.num_entity,
                                           self.option.relation_embed_size,
                                           padding_idx=self.data_loader.kg.pad_token_id
                                           )
        self.non_bert_parameters = [par for par in self.item_embedding.parameters()]

        # load bert if neccessary during training or evaluation
        if (option.reward == "context") or (option.metric == "context"):
            if option.mode.startswith("coke"):  ## hardcode to use CoKE
                self.path_scoring_model = CoKEWrapper(self.option.coke_mode, self.option.coke_config, self.option.coke_model,
                        self.option.dataset, self.option.coke_len, self.option.mask_head)
                self.embed_path = self.embed_coke_path
            else:
                self.embed_path = self.embed_bert_path
                if option.load_config:
                    self.path_scoring_model = BertForMaskedLM(config=BertConfig.from_pretrained(self.option.bert_path))
                else:
                    self.path_scoring_model = BertForMaskedLM.from_pretrained(self.option.bert_path)  # coke decomment

                self.make_bert_trainable()  # coke decomment
        else:
            ## replace the upper bert loading with this dummy function for debugging
            # self.path_scoring_model = self.fct
            self.path_scoring_model = torch.nn.Linear(1,2)

        # configure the learnable agent parameters
        if self.option.mode.startswith("bert"):
            self.policy_step = Bert_policy(self.data_loader, self.path_scoring_model, option)
            self.non_bert_parameters.extend([par for par in self.policy_step.to_state.parameters()])
        else:
            self.policy_step = Policy_step(self.option)
            self.non_bert_parameters.extend([par for par in self.policy_step.parameters()])
        if self.option.mode.endswith("mlp"):
            self.policy_mlp = Policy_mlp(self.option)
            self.non_bert_parameters.extend([par for par in self.policy_mlp.parameters()])

        self.state = None

        # control random state
        if self.option.use_cuda:
            self.generator = torch.Generator(device='cuda')
        else:
            self.generator = torch.Generator()
        
        self.generator = self.generator.manual_seed(self.option.random_seed)
        torch.manual_seed(self.option.random_seed)

        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)

    def make_bert_trainable(self):
        self.path_scoring_model.train(self.option.train_layers == [])
        for name, par in self.path_scoring_model.named_parameters():
            if any([f'layer.{id}' in name for id in self.option.train_layers]) or 'cls' in name and self.option.train_layers != []:
                par.requires_grad_(True)
                print(f"Layer {name} - activate training")
            else:
                par.requires_grad_(False)
                print(name) 
        print(self.path_scoring_model)

    def fct(self, seq):
        # only neccessary for testing purposes, simulates random bert output
        embs = torch.randn(seq.shape[0], seq.shape[1], 256, requires_grad=True)
        probs = torch.randn(seq.shape[0], seq.shape[1], self.option.num_entity + self.option.num_relation, requires_grad=True)
        return probs, embs

    def zero_state(self, dim):
        self.state = [torch.zeros(dim, self.option.state_embed_size).to(self.item_embedding.weight.device),
                      torch.zeros(dim, self.option.state_embed_size).to(self.item_embedding.weight.device)]

    def action_encoder(self, rel_ids, ent_ids):
        if self.option.use_entity_embed:
            parts = self.item_embedding(rel_ids), self.item_embedding(ent_ids)
            return torch.cat(parts, dim=-1)
        else:
            return self.item_embedding(rel_ids)

    def embed_coke_path(self, sequences, use_labels=None):
        # todo: check if instead of flipping relations should be inverted
        e0 = sequences[:, :1]
        path_rel = sequences[:, 3::2]
        qr = sequences[:, 1:2]
        t = sequences[:, 2:3]
        e1 = sequences[:, -1:]
        #print(sequences.shape, e0.shape, rel.shape)
        assert(path_rel.shape[1] == self.option.max_step_length)

        # flip to make tail prediction
        if self.option.mask_head:
            coke_seq = torch.cat([e0, rel, e1], dim=-1)
        else:
            rel_flip = torch.LongTensor([[self.data_loader.kg.rel2inv[r.item()] if r.item() not in [0,3]  else r.item() for r in r_seq] for r_seq in path_rel.flip(-1)])
            qr_flip = torch.LongTensor([[self.data_loader.kg.rel2inv[r.item()]] for r in qr])
            if self.option.coke_mode == "lp":
                coke_seq = torch.cat([rel_flip, t, qr_flip, e0], dim=-1)
            elif self.option.coke_mode == "anchor":
                coke_seq = torch.cat([e1, rel_flip, t, qr_flip, e0], dim=-1)
            elif self.option.coke_mode == "pqa":
                coke_seq = torch.cat([e1, rel_flip, qr_flip, e0], dim=-1)

        scores_np = self.path_scoring_model.get_predictions(coke_seq.detach().cpu().numpy())
        #print(scores_np.shape, self.item_embedding.num_embeddings)
        assert scores_np.shape[0] == sequences.shape[0]
        #assert scores_np.shape[1] == self.item_embedding.num_embeddings ## item embedding has extra "out of bert" tokens in kg_rl.py
        return 0, torch.from_numpy(scores_np)


    def embed_bert_path(self, sequences, use_labels=False):
        drop_tokens = self.option.token_droprate != 0.0

        device = next(self.path_scoring_model.parameters()).device
        if not self.test_mode and drop_tokens:
            dropout_mask = torch.bernoulli(torch.ones_like(sequences) * self.option.token_droprate).long()
            sequences[dropout_mask == 1] = self.data_loader.kg.unk_token_id
            # print((prev_action_embedding == self.data_loader.kg.unk_token_id).float().mean())

        cls = torch.ones(sequences.shape[0], 1).type(torch.int64) * self.data_loader.kg.cls_token_id
        sep = torch.ones(sequences.shape[0], 1).type(torch.int64) * self.data_loader.kg.sep_token_id
        inputs = copy.deepcopy(sequences).cpu()
        inputs[:, 0] = self.data_loader.kg.mask_token_id
        inputs = torch.cat((cls, inputs, sep), dim=-1).type(torch.int64).to(device)

        word_embeddings = self.path_scoring_model.bert.embeddings.word_embeddings(inputs.type(torch.int64))

        if self.test_mode and drop_tokens:
            word_embeddings *= 1 - self.token_droprate

        if use_labels:
            labels = torch.ones_like(inputs) * -1
            labels[:, 1] = sequences[:, 0]
            labels = labels.to(device)

            loss, probs, _ = self.path_scoring_model(inputs_embeds=word_embeddings,
                                                      masked_lm_labels=labels.type(torch.int64))
            return loss, probs[:,1,:]
        else:
            _, embs = self.path_scoring_model(inputs_embeds=word_embeddings)  # probs, embs
            return embs.cpu()

    def get_action_dist(self, prev_relation, current_entity, actions_id, queries, sequences):
        # Get state vector
        out_relations_id = actions_id[:, :, 0]  # B x n_actions
        out_entities_id = actions_id[:, :, 1]  # B x n_actions

        if self.option.random:
            prelim_scores = torch.randn(out_relations_id.shape)  # B x n_actions
            prelim_scores = prelim_scores.to(self.item_embedding.weight.device)
        elif self.option.mode.endswith("mlp"):
            action = self.action_encoder(out_relations_id, out_entities_id)  # B x n_actions x action_emb
            action = self.dropout(action)

            if self.option.mode == "bert_mlp":
                prev_action_embedding= self.embed_path(sequences.clone(), use_labels=False)
            else:
                prev_action_embedding = self.action_encoder(prev_relation, current_entity) # B x action_emb
                prev_action_embedding = self.dropout(prev_action_embedding)

            # 1. one step of rnn
            current_state, self.state = self.policy_step(prev_action_embedding, self.state)
            current_state = self.dropout(current_state)
            state_query = self.get_decision_input(queries,  current_state,
                                                  current_entity)

            # MLP for policy#
            output = self.policy_mlp(state_query)
            output = self.dropout(output)# B x 1 x action_emb
            prelim_scores = torch.sum(torch.mul(output, action), dim=-1)  # B x n_actions

        # Masking PAD actions
        dummy_actions_id = torch.ones_like(out_entities_id, dtype=torch.int64) * self.data_loader.kg.pad_token_id  # B x n_actions
        mask = torch.eq(out_entities_id, dummy_actions_id)  # B x n_actions
        dummy_scores = torch.ones_like(prelim_scores) * (-99999)  # B x n_actions
        scores = torch.where(mask, dummy_scores, prelim_scores)  # B x n_actions
        logits = scores.log_softmax(dim=-1)  # B x n_actions

        return logits, out_relations_id, out_entities_id

    def get_decision_input(self, queries, current_state, current_entity):
        current_state = current_state.to(queries.device)
        ent_q = self.action_encoder(queries, current_entity)
        return torch.cat([current_state, ent_q], -1)

    def step(self, *params):
        logits, out_relations_id, out_entities_id = self.get_action_dist(*params)

        # 4 sample action (epsilon-greedy)
        if self.option.epsilon > 0:
            non_greedy_mask = torch.rand(logits.shape[0], generator=self.generator, device='cuda') < self.option.epsilon
            action_id = torch.zeros_like(logits, dtype=torch.long)

            # set non-greedy logits to uniform
            uniform_logits = logits[non_greedy_mask].detach().clone()
            uniform_logits[uniform_logits.exp() > 1e-5] = 100  # set all non-zero (pad) positions   
            uniform_logits = uniform_logits.log_softmax(dim=-1)
            
            # replace original logits
            logits_to_sample = logits.detach().clone()
            logits_to_sample[non_greedy_mask] = uniform_logits
        else:
            logits_to_sample = logits.detach().clone()

        # sample from modified score distribution
        action_id = torch.multinomial(input=logits_to_sample.exp(), num_samples=1, generator=self.generator)  # B_greedy x 1

        # loss # lookup tf.nn.sparse_softmax_cross_entropy_with_logits
        # 5a.
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)  # B x n_actions
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)  # B x n_actions

        # 6. Map back to true id
        chosen_relation = torch.gather(out_relations_id, dim=1, index=action_id).squeeze()
        next_entities = torch.gather(out_entities_id, dim=1, index=action_id).squeeze()
        action_id = action_id.squeeze()
        # assert (next_entities == self.graph.get_next(current_entities, action_id)).all()

        #sss = self.data_loader.num2relation[(int)(queries[0])] + "\t" + self.data_loader.num2relation[(int)(chosen_relation[0])]
        #log.info(sss)

        return loss, logits, action_id, next_entities, chosen_relation

    def test_step(self, prev_relation, current_entity, actions_id, log_current_prob, queries, batch_size,
                  sequences, step):

        log_action_prob, out_relations_id, out_entities_id = self.get_action_dist(
            prev_relation, current_entity, actions_id, queries, sequences)

        top_k_action_id, log_current_prob = self.test_search(log_current_prob, log_action_prob, batch_size, step)
        chosen_relation, chosen_entities, sequences = self.update_search_states(
            top_k_action_id,
            out_relations_id, out_entities_id, sequences, batch_size)

        return chosen_relation, chosen_entities, log_current_prob, sequences

    def test_search(self, log_current_prob, log_action_prob,
                    batch_size, step):
        ## tf: trainer beam search ##

        ## CAREFUL: t=torch.arange(6); t.view(3,2) does not equal t.view(2,3).t()
        # be aware of the flattened order of the elements

        # shape: BATCH*TIMES --> BATCH x TIMES*MAX_OUT
        # linear order: BATCH(TIMES(MAX_OUT)))

        #print("current, action", log_current_prob.shape, log_action_prob.shape)
        log_current_prob = log_current_prob.repeat_interleave(self.option.max_out).view(batch_size, -1)
        log_action_prob = log_action_prob.view(batch_size, -1)
        #print("current, action", log_current_prob.shape, log_action_prob.shape)
        log_trail_prob = torch.add(log_action_prob, log_current_prob)
        if (step != self.option.max_step_length -1) or self.option.reward == "answer":
            top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.option.test_times)  # B x TIMES
            # action ids in range 0, TIMES*MAX_OUT
        else:
            # for the last step of the context generation: take only the most probable path
            top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, 1)
        log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)
        return top_k_action_id, log_current_prob

    def update_search_states(self, top_k_action_id, out_relations_id, out_entities_id, sequences, batch_size):
        if self.option.mode.endswith("mlp"):
            new_state_0 = self.state[0].unsqueeze(1)  # .repeat(1, self.option.max_out, 1)
            # change B*TIMES x MAX_OUT x STATE_DIM --> B x TIMES*MAX_OUT x STATE_DIM
            new_state_0 = self.state[0].view(batch_size, -1, self.option.state_embed_size)
            new_state_1 = self.state[1].unsqueeze(1)  #.repeat(1, self.option.max_out, 1)
            new_state_1 = self.state[1].view(batch_size, -1, self.option.state_embed_size)

            # select history according to beam search
            top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1,
                                                                        self.option.state_embed_size) // self.option.max_out
            self.state = \
                (torch.gather(new_state_0, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size),
                 torch.gather(new_state_1, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size))

        # B*TIMES x MAX_OUT --> B x TIMES*MAX_OUT
        out_relations_id = out_relations_id.view(batch_size, -1)
        out_entities_id = out_entities_id.view(batch_size, -1)

        # select action according to beam search
        chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
        chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
        # assert (log_current_prob == top_k_log_prob.view(-1)).all()

        # select relevant sequences according to beam search
        seq_len = sequences.shape[-1]
        top_k_action_id_seq = top_k_action_id.unsqueeze(2).repeat(1, 1, seq_len) // self.option.max_out
        # B * times x seq_len --> B x times x seq_len
        sequences = sequences.unsqueeze(1).view(batch_size, -1, seq_len)

        sequences = torch.gather(sequences, dim=1, index=top_k_action_id_seq).view(-1, seq_len)

        # append new elements to the sequences
        sequences = torch.cat((sequences, chosen_relation.view(-1, 1), chosen_entities.view(-1, 1)), dim=-1)
        #sequences = sequences.view(batch_size, -1, sequences.shape[-1])

        return chosen_relation, chosen_entities, sequences

    def get_dummy_start_relation(self, batch_size):
        dummy_start_item = self.data_loader.relation2num['START']
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start

    def get_reward(self, current_entities, answers, positive_reward, negative_reward):
        reward = (current_entities == answers)
        reward = torch.where(reward, positive_reward, negative_reward)
        return reward

    def get_context_reward(self, sequences, all_correct, metric=1):
        
        # print("seq to reward: ", sequences)
        loss, scores = self.embed_path(sequences, use_labels=True)

        labels = sequences[:,0].numpy().reshape(-1)
        #print(scores.shape)
        prediction_prob = scores.softmax(dim=-1)  #.detach().numpy()  # B x n_actions
        
        #print(prediction_prob.shape)
        rewards_prob = prediction_prob[np.arange(prediction_prob.shape[0]), labels] # B x 1
        if not self.test_mode:
            # for unfiltered rank == 1 uncomment:
            # rewards_prob = rewards_prob > np.percentile(prediction_prob, q=99.9, axis=-1)
            #times = self.option.train_times
            #mean_per_episode = rewards_prob.reshape(-1, times).mean(-1, keepdim=True) # B
            #print(mean_per_episode.shape, rewards_prob.shape)
            #rewards_prob = (rewards_prob.reshape(-1, times) - mean_per_episode).reshape(-1) #.clamp_min_(0)
            #loss = loss * rewards_prob.detach().cpu().mean().clamp_min_(0) 
            return loss, rewards_prob, None

        rewards_rank = np.empty_like(labels).astype(np.float)
        ranks = np.empty_like(labels).astype(np.int)

        ranked_token_ids = torch.argsort(scores.detach().cpu(), descending=True, dim=-1).numpy()

        for i, label in enumerate(labels.tolist()):
            ranked = ranked_token_ids[i].tolist()
            ranked = [x for x in ranked if ((x not in all_correct[i] and x < self.option.num_entity and x>4) or x == label)]
            rank = ranked.index(label)

            ranks[i] = rank
            if rank < metric:
                rewards_rank[i] = 1
            else:
                rewards_rank[i] = 0
        return rewards_rank, rewards_prob, ranks

    def print_parameter(self):
        for param in self.named_parameters():
            print(param[0], param[1])

    def my_state_dict(self):
        return {k:v for k,v in self.state_dict().items()} # if not k.startswith('path_scoring_model')}

