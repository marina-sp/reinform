import os
from collections import defaultdict
import numpy as np
import copy

from kg_rl import *

class Data_loader():
    def __init__(self, option):
        self.option = option
        self.include_reverse = option.use_inverse

        self.entity2num = None
        self.num2entity = None

        self.relation2num = None
        self.num2relation = None
        self.relation2inv = None

        self.num_relation = 0
        self.num_entity = 0
        self.num_operator = 0

        data_path = os.path.join(self.option.datadir, self.option.dataset)

        self.kg = CustomKG("freebase15k_237")
        self.kg.prepare_data()
        self.kg.add_reversed_relations()
        self.kg.add_extra_relations()

        self.load_data_all(data_path)

    def load_data_all(self, path):
        self.entity2num, self.num2entity = self.kg.entity2idx, self.kg.idx2entity
        self.relation2num, self.num2relation = self.kg.relation2idx, self.kg.idx2relation
        # todo
        # assert len(self.entity2num) == len(self.num2entity)
        # assert len(self.relation2num) == len(self.num2relation)

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num) + self.kg.reserved_vocab
        print("num_relation", self.num_relation)
        print("num_entity", self.num_entity)

        self.data = {}
        # todo: check inverse relations in data
        self.data['train'], self.data['inv_train'] = self.load_data("train"), []
        self.data['valid'], self.data['inv_valid'] = self.load_data("valid"), []
        self.data['test'],  self.data['inv_test']  = self.load_data("test"), []

        # todo
        # print("total seen relations", len(self.relation2num) - 3)
        # print("total seen entities", len(self.entity2num) - 1)

    def load_data(self, data):
        return [[t.h, t.r, t.t] for t in self.kg.triplets[data]]

    def get_data(self, data):
        out = self.data['data'] + self.data[f'inv_{data}'] if self.include_reverse else self.data[data]
        with open(os.path.join(self.option.this_expsdir, f"{data}_log.txt"), "a+", encoding='UTF-8') as f:
            f.write(f"{data} data contains {len(out)} triples\n")
        return np.array(out, dtype=np.int64)

    def get_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Graph contains " + str(len(self.data['train'] + self.data['inv_train'])) + " triples\n")
        return np.array(self.data['train'] + self.data['inv_train'], dtype=np.int64)
