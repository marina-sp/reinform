import torch
import numpy as np
from collections import defaultdict
import copy


class Knowledge_graph():
    def __init__(self, option, data_loader, data):
        self.option = option
        self.data = data
        self.data_loader = data_loader
        self.out_array = None
        self.all_correct = None
        self.construct_graph()

    # 根据原始数据构建知识图谱，out_array存储每个节点向外的出口路径数组
    def construct_graph(self):
        all_out_dict = defaultdict(list)
        for head, relation, tail, _ in self.data:
            all_out_dict[head].append((relation, tail))

        all_correct = defaultdict(set)

        # out array: batch size x max actions x 2 (entity, relation)
        out_array = np.ones((self.option.num_entity, self.option.max_out, 2), dtype=np.int64)
        out_array[:, :, 0] *= self.data_loader.kg.pad_token_id
        out_array[:, :, 1] *= self.data_loader.kg.pad_token_id
        more_out_count = 0
        for head in all_out_dict:
            # 1st action reserved for: stay in the same state
            if self.option.reward == "answer":
                out_array[head, 0, 0] = self.data_loader.kg.unk_token_id
                out_array[head, 0, 1] = head
                num_out = 1
            else:
                # add action to "END" state from every entity
                num_out = 0
            for relation, tail in all_out_dict[head]:
                if num_out == self.option.max_out:
                    more_out_count += 1
                    break
                out_array[head, num_out, 0] = relation
                out_array[head, num_out, 1] = tail
                num_out += 1
                all_correct[(head, relation)].add(tail)
        self.out_array = torch.from_numpy(out_array)
        self.all_correct = all_correct
        print("more_out_count", more_out_count)

    # 获取从图谱上current_entities的out_relations, out_entities
    def get_out(self, current_entities, start_entities, query_relations, answers, all_correct, step):
        ret = copy.deepcopy(self.out_array[current_entities, :, :])
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]

                if self.option.reward == "context":
                    # note: different from the orig due to front masking (mask inverse triple)
                    mask = (self.data_loader.kg.rel2inv[query_relations[i].item()] == relations)
                elif self.option.reward == "answer":
                    # orig masking
                    mask = (query_relations[i] == relations)
                mask = mask & answers[i].eq(entities)

                ret[i, :, 0][mask] = self.data_loader.kg.pad_token_id
                ret[i, :, 1][mask] = self.data_loader.kg.pad_token_id
            elif current_entities[i] == answers[i]:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]
                if self.option.reward == "context":
                    # note: different from the orig due to front masking (mask inverse triple)
                    mask = (query_relations[i] == relations)
                elif self.option.reward == "answer":
                    # orig masking
                    mask = (self.data_loader.kg.rel2inv[query_relations[i].item()] == relations)
                mask = mask & start_entities[i].eq(entities)
                ret[i, :, 0][mask] = self.data_loader.kg.pad_token_id
                ret[i, :, 1][mask] = self.data_loader.kg.pad_token_id

            if self.option.reward == "answer":
                # filter other correct answers at the last step
                if step == (self.option.max_step_length - 1):
                    relations = ret[i, :, 0]
                    entities = ret[i, :, 1]
                    answer = answers[i]
                    for j in range(entities.shape[0]):
                        if entities[j].item() in all_correct[i] and entities[j] != answer:
                            relations[j] = self.data_loader.kg.pad_token_id
                            entities[j] = self.data_loader.kg.pad_token_id

        return ret

    def get_all_correct(self, start_entities_np, relations_np):
        all_correct = list()
        for i in range(start_entities_np.shape[0]):
            all_correct.append(self.all_correct[(start_entities_np[i], relations_np[i])])
        return all_correct

    def update_all_correct(self, data):
        for head, relation, tail, inv_relation in data:
            self.all_correct[(head, relation)].add(tail)
            self.all_correct[(tail, inv_relation)].add(head)
