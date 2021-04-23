import os
from collections import defaultdict
import numpy as np
import copy

from kg_rl import *

class Data_loader():
    def __init__(self, option):
        self.option = option
        self.include_reverse = option.use_inverse

        data_path = os.path.join(self.option.datadir, self.option.dataset)

        self.kg = CustomKG(dataset=self.option.dataset) # "freebase15k_237" "WN18_RR"
        self.kg.prepare_data()
        self.kg.add_reversed_relations()
        self.kg.add_extra_relations()
        print(self.kg.rel2inv)

        self.load_mappings()
        self.load_data_all(data_path)

    def load_mappings(self):
        self.entity2num, self.num2entity = self.kg.entity2idx, self.kg.idx2entity
        self.relation2num, self.num2relation = self.kg.relation2idx, self.kg.idx2relation
        self.num2relation[self.kg.unk_token_id] = "NO_OP"
        self.num2relation[self.kg.pad_token_id] = "PAD"
        self.num2entity[self.kg.pad_token_id] = "PAD"
        self.num2entity[self.kg.unk_token_id] = "UNK"
        # self._augment_reverse_relation()
        # self._add_item(self.relation2num, self.num2relation, "Equal")
        # self._add_item(self.relation2num, self.num2relation, "Pad")
        # self._add_item(self.relation2num, self.num2relation, "Start")
        # self._add_item(self.entity2num, self.num2entity, "Pad")
        # print(self.relation2num)

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num) + self.kg.reserved_vocab
        print("num_relation", self.num_relation)
        print("num_entity", self.num_entity - self.kg.reserved_vocab)

    ## finilize for different relations and entity embedding sizes
    #     # map joint Transformer indices for relations and entities to separate mappings
    #     self.mixed2ent = {v: i for i, v in enumerate(self.kg.entity2idx.values())}
    #     self.ent2mixed = {v: k for k, v in self.mixed2ent.items()}
    #     # include reserved relations
    #     self.mixed2rel = {v: i for i, v in enumerate(self.kg.entity2idx.values())}
    #     self.rel2mixed = {v: k for k, v in self.mixed2rel.items()}
    #     # add pad token mapping
    #     self._add_item(self.mixed2rel, self.rel2mixed, 0)
    #
    #     # include reserved tokens for embedding
    #     self.num_relation = len(self.mixed2ent)
    #     self.num_entity = len(self.mixed2rel)
    #     # print data stats without reserved tokens
    #     print("num_relation", self.num_relation)
    #     print("num_entity", self.num_entity)
    #
    #     # assert len(self.entity2num) == len(self.num2entity)
    #     # assert len(self.relation2num) == len(self.num2relation)
    #
    # def _add_item(self, obj2num, num2obj, item):
    #     count = len(obj2num)
    #     obj2num[item] = count
    #     num2obj[count] = item
    #
    #

    def load_data_all(self, path):
        self.data = {}

        self.data['train'], self.data['inv_train'] = self.load_data("train")
        self.data['valid'], self.data['inv_valid'] = self.load_data("valid")
        self.data['test'],  self.data['inv_test']  = self.load_data("test")


    def load_data(self, data):
        ## inversed triplets in kg are added last
        split = len(self.kg.triplets[data]) // 2
        assert split * 2 == len(self.kg.triplets[data])
        hrt = [[t.h, t.r, t.t, self.kg.rel2inv[t.r]] for t in self.kg.triplets[data][:split]]
        trh = [[t.h, t.r, t.t, self.kg.rel2inv[t.r]] for t in self.kg.triplets[data][split:]]
        assert np.array(hrt)[:,1].max() < np.array(trh)[:,1].min()  # indices of inverse relations must be higher
        return (hrt, trh)

    def get_data(self, data, mode='eval'):
        if mode == 'train':
            out = self.data[data] + self.data[f'inv_{data}'] if self.include_reverse else self.data[data]
        else:
            out = self.data[data] + self.data[f'inv_{data}']  
        with open(os.path.join(self.option.this_expsdir, f"{data}_log.txt"), "a+", encoding='UTF-8') as f:
            f.write(f"{data} data contains {len(out)} triples\n")
        return np.array(out, dtype=np.int64)

    def get_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Graph contains " + str(len(self.data['train'] + self.data['inv_train'])) + " triples\n")
        return np.array(self.data['train'] + self.data['inv_train'], dtype=np.int64)
