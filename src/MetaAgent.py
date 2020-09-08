import torch.nn as nn
import numpy as np
import torch
import logging as log
from torch.distributions.categorical import Categorical
from collections import defaultdict
import copy
from transformers import BertForMaskedLM, BertConfig
from Agent import Policy_step, Policy_mlp, Bert_policy, Agent

class MetaAgent(Agent):
    def __init__(self, option, data_loader, graph=None):
        super(MetaAgent, self).__init__(option, data_loader)

        # add
        self.policy_mlp.mlp_l2 = nn.Linear(self.option.mlp_hidden_size, 
                self.option.action_embed_size +1, bias=True)

        # replace with dumped option reading
        option2 = copy.deepcopy(option)
        option2.__dict__.update(
            {'mode': 'lstm_mlp',
             'action_embed_size': 100,
             'max_step_length': 3,
             'metric': 'answer',
             'mlp_hidden_size': 200,
             'relation_embed_size': 100,
             'reward': 'answer',
             'state_embed_size': 200,
             'test_times': 100,
             'use_entity_embed': False,
             'use_inverse': False}
        )
        self.advisor = Agent(option2, data_loader)
        self.advisor.eval()
        for par in self.advisor.parameters():
            par.requires_grad_(False)

    def action_encoder(self, rel_ids, ent_ids, meta=True):
        if self.option.use_entity_embed:
            parts = self.item_embedding(rel_ids), self.item_embedding(ent_ids)
            action = torch.cat(parts, dim=-1)
        else:
            action = self.item_embedding(rel_ids)
        if meta and self.advisor_logits is not None:
            print(self.advisor_logits.shape)
            action = torch.cat([action, self.advisor_logits.unsqueeze(-1)], dim=-1)
            print(action.shape)
            self.advisor_logits = None
        return action

    def get_decision_input(self, queries, current_state, current_entity):
        current_state = current_state.to(queries.device)
        ent_q = self.action_encoder(queries, current_entity, meta=False)
        return torch.cat([current_state, ent_q], -1)

    def get_action_dist(self, *params):
        self.advisor_logits, _, _ = self.advisor.get_action_dist(*params)
        return super().get_action_dist(*params)
    
    def zero_state(self, dim):
        self.advisor.zero_state(dim)
        super().zero_state(dim)

    # def test_step(self, prev_relation, current_entity, actions_id, log_current_prob, queries, batch_size,
    #               sequences, step, random):
    #
    #     log_action_prob, out_relations_id, out_entities_id = self.get_action_dist(
    #         prev_relation, current_entity, actions_id, queries, sequences, random)
    #
    #     chosen_relation, chosen_entities, log_current_prob, sequences = self.test_search(
    #         log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size,
    #          sequences, step)
    #
    #     return chosen_relation, chosen_entities, log_current_prob, sequences
    #
    # def test_search(self, log_current_prob, log_action_prob, out_relations_id, out_entities_id,
    #                 batch_size, sequences, step):
    #     ## tf: trainer beam search ##
    #
    #     ## CAREFUL: t=torch.arange(6); t.view(3,2) does not equal t.view(2,3).t()
    #     # be aware of the flattened order of the elements
    #
    #     # shape: BATCH*TIMES --> BATCH x TIMES*MAX_OUT
    #     # linear order: BATCH(TIMES(MAX_OUT)))
    #
    #     log_current_prob = log_current_prob.repeat_interleave(self.option.max_out).view(batch_size, -1)
    #     log_action_prob = log_action_prob.view(batch_size, -1)
    #     log_trail_prob = torch.add(log_action_prob, log_current_prob)
    #     if (step != self.option.max_step_length -1) or self.option.reward == "answer":
    #         top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.option.test_times)  # B x TIMES
    #         # action ids in range 0, TIMES*MAX_OUT
    #     else:
    #         # for the last step of the context generation: take only the most probable path
    #         top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, 1)
    #
    #     if self.option.mode == "lstm_mlp":
    #         new_state_0 = self.state[0].unsqueeze(1)  # .repeat(1, self.option.max_out, 1)
    #         # change B*TIMES x MAX_OUT x STATE_DIM --> B x TIMES*MAX_OUT x STATE_DIM
    #         new_state_0 = self.state[0].view(batch_size, -1, self.option.state_embed_size)
    #         new_state_1 = self.state[1].unsqueeze(1)  #.repeat(1, self.option.max_out, 1)
    #         new_state_1 = self.state[1].view(batch_size, -1, self.option.state_embed_size)
    #
    #         # select history according to beam search
    #         top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1,
    #                                                                     self.option.state_embed_size) // self.option.max_out
    #         self.state = \
    #             (torch.gather(new_state_0, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size),
    #              torch.gather(new_state_1, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size))
    #
    #     # B*TIMES x MAX_OUT --> B x TIMES*MAX_OUT
    #     out_relations_id = out_relations_id.view(batch_size, -1)
    #     out_entities_id = out_entities_id.view(batch_size, -1)
    #
    #     # select action according to beam search
    #     chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
    #     chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
    #     log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)
    #     # assert (log_current_prob == top_k_log_prob.view(-1)).all()
    #
    #     # select relevant sequences according to beam search
    #     seq_len = sequences.shape[-1]
    #     top_k_action_id_seq = top_k_action_id.unsqueeze(2).repeat(1, 1, seq_len) // self.option.max_out
    #     # B * times x seq_len --> B x times x seq_len
    #     sequences = sequences.unsqueeze(1).view(batch_size, -1, seq_len)
    #
    #     sequences = torch.gather(sequences, dim=1, index=top_k_action_id_seq).view(-1, seq_len)
    #
    #     # append new elements to the sequences
    #     sequences = torch.cat((sequences, chosen_relation.view(-1, 1), chosen_entities.view(-1, 1)), dim=-1)
    #     #sequences = sequences.view(batch_size, -1, sequences.shape[-1])
    #
    #     return chosen_relation, chosen_entities, log_current_prob, sequences
    #
    # def get_dummy_start_relation(self, batch_size):
    #     dummy_start_item = self.data_loader.relation2num['START']
    #     dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
    #     return dummy_start
    #
    # def get_reward(self, current_entities, answers, positive_reward, negative_reward):
    #     reward = (current_entities == answers)
    #     reward = torch.where(reward, positive_reward, negative_reward)
    #     return reward
    #
    # def get_context_reward(self, sequences, all_correct, metric=1, test=False):
    #
    #     inputs = copy.deepcopy(sequences)
    #     inputs[:,0] = self.data_loader.kg.mask_token_id
    #     cls_tensor = torch.ones((inputs.size(0),), dtype=torch.int8)*self.data_loader.kg.cls_token_id
    #     sep_tensor = torch.ones((inputs.size(0),), dtype=torch.int8)*self.data_loader.kg.sep_token_id
    #     inputs = inputs.type(torch.IntTensor)
    #     cls_tensor = cls_tensor.type(torch.IntTensor)
    #     sep_tensor = sep_tensor.type(torch.IntTensor)
    #     inputs = torch.cat((cls_tensor.reshape((cls_tensor.shape[0],-1)),inputs, sep_tensor.reshape((sep_tensor.shape[0],-1))),1)
    #     #labels = sequences[:,0].numpy().reshape(-1)
    #     labels = torch.ones_like(inputs) * -1
    #     labels[:, 1] = sequences[:, 0]
    #     if next(self.path_scoring_model.parameters()).device.type == 'cuda':
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #
    #     loss, output, _ = self.path_scoring_model(inputs.type(torch.int64), masked_lm_labels=labels.type(torch.int64))
    #
    #     labels = sequences[:,0].numpy().reshape(-1)
    #     prediction_scores = output[:, 1].cpu()
    #     prediction_prob = prediction_scores.softmax(dim=-1)  #.detach().numpy()  # B x n_actions
    #
    #     rewards_prob = prediction_prob[np.arange(prediction_prob.shape[0]), labels]
    #     if not test:
    #         # for unfiltered rank == 1 uncomment:
    #         # rewards_prob = rewards_prob > np.percentile(prediction_prob, q=99.9, axis=-1)
    #         return loss, rewards_prob, None
    #
    #     rewards_rank = np.empty_like(labels).astype(np.float)
    #     ranks = np.empty_like(labels).astype(np.int)
    #
    #     ranked_token_ids = torch.argsort(prediction_scores, descending=True, dim=-1).numpy()
    #
    #     for i, label in enumerate(labels.tolist()):
    #         ranked = ranked_token_ids[i].tolist()
    #         ranked = [x for x in ranked if ((x not in all_correct[i] and x < self.option.num_entity) or x == label)]
    #         rank = ranked.index(label)
    #
    #         ranks[i] = rank
    #         if rank < metric:
    #             rewards_rank[i] = 1
    #         else:
    #             rewards_rank[i] = 0
    #     return rewards_rank, rewards_prob, ranks
    #
    # def print_parameter(self):
    #     for param in self.named_parameters():
    #         print(param[0], param[1])
    #
    # def my_state_dict(self):
    #     return {k:v for k,v in self.state_dict().items()} # if not k.startswith('path_scoring_model')}

