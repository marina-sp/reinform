import torch.nn as nn
import numpy as np
import torch
import logging as log
from torch.distributions.categorical import Categorical
from collections import defaultdict
import copy
from transformers import BertForMaskedLM

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


class Agent(nn.Module):
    def __init__(self, option, data_loader, graph=None):
        super(Agent, self).__init__()
        self.option = option
        self.data_loader = data_loader
        # self.relation_embedding = nn.Embedding(self.option.num_relation, self.option.relation_embed_size)
        self.relation_embedding = nn.Embedding(self.option.num_relation + self.option.num_entity,
                                               self.option.relation_embed_size)
        self.policy_step = Policy_step(self.option)
        self.policy_mlp = Policy_mlp(self.option)

        # todo: load optionally
        self.path_scoring_model = BertForMaskedLM.from_pretrained(self.option.bert_path)
        self.path_scoring_model.eval()

        # control random state
        if self.option.use_cuda:
            self.generator = torch.cuda.manual_seed(self.option.random_seed)
            torch.manual_seed(self.option.random_seed)
        else:
            self.generator = torch.manual_seed(self.option.random_seed)

        torch.nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        if self.option.use_entity_embed:
            self.entity_embedding = nn.Embedding(self.option.num_entity, self.option.entity_embed_size)

    def action_encoder(self, rel_ids, ent_ids):
        if self.option.use_entity_embed:
            parts = self.relation_embedding(rel_ids), self.entity_embedding(ent_ids)
            return torch.cat(parts, dim=-1)
        else:
            return self.relation_embedding(rel_ids)

    def _step(self, prev_state, prev_relation, current_entity, actions_id, queries):
        prev_action_embedding = self.action_encoder(prev_relation, current_entity)
        # 1. one step of rnn
        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        # Get state vector
        out_relations_id = actions_id[:, :, 0]  # B x n_actions
        out_entities_id = actions_id[:, :, 1]  # B x n_actions
        action = self.action_encoder(out_relations_id, out_entities_id)  # B x n_actions x action_emb

        current_state = output.squeeze()
        queries_embedding = self.relation_embedding(queries)
        if self.option.use_entity_embed:
            entity_embedding = self.entity_embedding(current_entity)
            state_query = torch.cat([current_state, queries_embedding, entity_embedding], -1)
        else:
            state_query = torch.cat([current_state, queries_embedding], -1)

        # MLP for policy#
        output = self.policy_mlp(state_query)  # B x 1 x action_emb
        prelim_scores = torch.sum(torch.mul(output, action), dim=-1)  # B x n_actions

        # Masking PAD actions
        dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.kg.pad_token_id  # B x n_actions
        mask = torch.eq(out_relations_id, dummy_relations_id)  # B x n_actions
        dummy_scores = torch.ones_like(prelim_scores) * (-99999)  # B x n_actions
        scores = torch.where(mask, dummy_scores, prelim_scores)  # B x n_actions
        logits = scores.log_softmax(dim=-1)  # B x n_actions

        return logits, out_relations_id, out_entities_id, new_state

    def step(self, *params):
        logits, out_relations_id, out_entities_id, new_state = self._step(*params)

        # 4 sample action
        action_id = torch.multinomial(input=logits.exp(), num_samples=1, generator=self.generator)  # B x 1
        #action_id = Categorical(logits=logits).sample((1,)).view(-1, 1)  # B x 1

        # print(" ".join(str(e.item()) for e in action_id.flatten())) # check random state

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

        return loss, new_state, logits, action_id, next_entities, chosen_relation

    def test_step(self, prev_state, prev_relation, current_entity, actions_id, log_current_prob, queries, batch_size,
                  sequences, last_step):

        log_action_prob, out_relations_id, out_entities_id, new_state = self._step(
            prev_state, prev_relation, current_entity, actions_id, queries)

        chosen_state, chosen_relation, chosen_entities, log_current_prob, sequences = self.test_search\
            (new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size,
             sequences, last_step)

        return chosen_state, chosen_relation, chosen_entities, log_current_prob, sequences

    def test_search(self, new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id,
                    batch_size, sequences, last_step):
        ## tf: trainer beam search ##

        ## CAREFUL: t=torch.arange(6); t.view(3,2) does not equal t.view(2,3).t()
        # be aware of the flattened order of the elements

        # shape: BATCH*TIMES --> BATCH x TIMES*MAX_OUT
        # linear order: BATCH(TIMES(MAX_OUT)))

        # todo: adjust for the normal search at the last step
        log_current_prob = log_current_prob.repeat_interleave(self.option.max_out).view(batch_size, -1)
        log_action_prob = log_action_prob.view(batch_size, -1)
        log_trail_prob = torch.add(log_action_prob, log_current_prob)
        if not last_step:
            top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.option.test_times)  # B x TIMES
        else:
            top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, 1)

        # copy
        new_state_0 = new_state[0].unsqueeze(1).repeat(1, self.option.max_out, 1)
        # change B*TIMES x MAX_OUT x STATE_DIM --> B x TIMES*MAX_OUT x STATE_DIM
        new_state_0 = new_state_0.view(batch_size, -1, self.option.state_embed_size)
        new_state_1 = new_state[1].unsqueeze(1).repeat(1, self.option.max_out, 1)
        new_state_1 = new_state_1.view(batch_size, -1, self.option.state_embed_size)

        # B*TIMES x MAX_OUT --> B x TIMES*MAX_OUT
        out_relations_id = out_relations_id.view(batch_size, -1)
        out_entities_id = out_entities_id.view(batch_size, -1)

        # select action according to beam search
        chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
        chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
        log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)
        # assert (log_current_prob == top_k_log_prob.view(-1)).all()

        # select history according to beam search
        top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1, self.option.state_embed_size)
        chosen_state = \
            (torch.gather(new_state_0, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size),
             torch.gather(new_state_1, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size))

        top_k_action_id_seq = top_k_action_id.unsqueeze(2).repeat(1, 1, sequences.shape[-1]) // self.option.max_out
        sequences = torch.gather(sequences, dim=1, index=top_k_action_id_seq).view(-1, sequences.shape[-1])

        # append new elements to the sequences
        sequences = torch.cat((sequences, chosen_relation.view(-1, 1), chosen_entities.view(-1, 1)), dim=-1)
        sequences = sequences.view(batch_size, -1, sequences.shape[-1])

        return chosen_state, chosen_relation, chosen_entities, log_current_prob, sequences

    def get_dummy_start_relation(self, batch_size):
        dummy_start_item = self.data_loader.kg.pad_token_id
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start

    def get_reward(self, current_entities, answers, positive_reward, negative_reward):
        reward = (current_entities == answers)
        reward = torch.where(reward, positive_reward, negative_reward)
        return reward

    def get_context_reward(self, sequences, all_correct, metric=1, test=False):

        inputs = copy.deepcopy(sequences)
        inputs[:,0] = self.data_loader.kg.mask_token_id
        cls_tensor = torch.ones((inputs.size(0),), dtype=torch.int8)*self.data_loader.kg.cls_token_id
        sep_tensor = torch.ones((inputs.size(0),), dtype=torch.int8)*self.data_loader.kg.sep_token_id
        inputs = inputs.type(torch.IntTensor)
        cls_tensor = cls_tensor.type(torch.IntTensor)
        sep_tensor = sep_tensor.type(torch.IntTensor)
        inputs = torch.cat((cls_tensor.reshape((cls_tensor.shape[0],-1)),inputs, sep_tensor.reshape((sep_tensor.shape[0],-1))),1)
        labels = torch.ones_like(inputs)*-1
        labels[:,1]=sequences[:,0]
        if self.option.use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        _, output, _ = self.path_scoring_model(inputs.type(torch.int64), masked_lm_labels=labels.type(torch.int64))
        #output = torch.randn(inputs.shape[0], 9, len(self.data_loader.kg.vocab))
        prediction_scores, labels = output[:, 1].cpu(), labels[:, 1].cpu().numpy()
        prediction_prob = prediction_scores.softmax(dim=-1).detach().numpy()

        rewards_prob = prediction_prob[np.arange(prediction_prob.shape[0]), labels]

        if not test:
            return None, rewards_prob, None

        rewards_rank = np.empty_like(labels).astype(np.int)
        ranks = np.empty_like(labels).astype(np.int)

        ranked_token_ids = torch.argsort(prediction_scores, descending=True, dim=-1).numpy()

        for i, label in enumerate(labels.tolist()):
            ranked = ranked_token_ids[i].tolist()
            ranked = [x for x in ranked if ((x not in all_correct[i] and x < self.option.num_entity) or x == label)]
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
