import os
from collections import defaultdict
import numpy as np
import copy

from kg_rl import *

class Data_loader():
    def __init__(self, option):
        self.option = option
        self.include_reverse = option.use_inverse

        self.train_data = None
        self.test_data = None
        self.valid_data = None

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
        # train_data_path = os.path.join(path, "train.txt")
        # test_data_path = os.path.join(path, "test.txt")
        # valid_data_path = os.path.join(path, "valid.txt")
        # entity_path = os.path.join(path, "entities.txt")
        # relations_path = os.path.join(path, "relations.txt")

        self.entity2num, self.num2entity = self.kg.entity2idx, self.kg.idx2entity
        self.relation2num, self.num2relation = self.kg.relation2idx, self.kg.idx2relation
        # todo
        # assert len(self.entity2num) == len(self.num2entity)
        # assert len(self.relation2num) == len(self.num2relation)
        # self._augment_reverse_relation()
        # self.num2relation[0] = "Unk"
        # self.num2entity[0] = "Unk"

        # self._augment_reverse_relation()
        # self._add_item(self.relation2num, self.num2relation, "Equal")
        # self._add_item(self.relation2num, self.num2relation, "Pad")
        # self._add_item(self.relation2num, self.num2relation, "Start")
        # self._add_item(self.entity2num, self.num2entity, "Pad")
        # print(self.relation2num)

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num) + self.kg.reserved_vocab
        print("num_relation", self.num_relation)
        print("num_entity", self.num_entity)

        # todo: check inverse relations in data
        self.train_data, self.inv_train_data = self.load_data("train"), []
        self.valid_data, self.inv_valid_data = self.load_data("valid"), []
        self.test_data, self.inv_test_data = self.load_data("test"), []

        # todo
        # print("total seen relations", len(self.relation2num) - 3)
        # print("total seen entities", len(self.entity2num) - 1)

    def load_data(self, data):
        return [[t.h, t.r, t.t] for t in self.kg.triplets[data]]

    def _load_data(self, path):
        data = [l.strip().split("\t") for l in open(path, "r").readlines()]
        triplets, inv_triplets = list(), list()
        for item in data:
            if item[0] not in self.entity2num or item[2] not in self.entity2num:
                pass  # print(path, " contains unknown entities")
            head = self.entity2num[item[0]]
            tail = self.entity2num[item[2]]
            relation = self.relation2num[item[1]]
            triplets.append([head, relation, tail])

            inv_relation = self.relation2num["inv_" + item[1]]
            inv_triplets.append([tail, inv_relation, head])
        return triplets, inv_triplets

    def _load_dict(self, path, first_num=1):
        obj2num = defaultdict(int)
        num2obj = dict()
        data = [l.strip() for l in open(path, "r").readlines()]
        for num, obj in enumerate(data):
            obj2num[obj] = first_num + num
            num2obj[first_num + num] = obj
        return obj2num, num2obj

    def _augment_reverse_relation(self):
        num_relation = len(self.num2relation)
        temp = list(self.num2relation.items())
        self.relation2inv = defaultdict(int)
        for n, r in temp:
            rel = "inv_" + r
            num = num_relation + n
            self.relation2num[rel] = num
            self.num2relation[num] = rel
            self.relation2inv[n] = num
            self.relation2inv[num] = n
        assert len(temp) * 2 == len(self.relation2inv) == len(self.relation2num) == len(self.num2relation)

    def _add_item(self, obj2num, num2obj, item):
        count = len(num2obj)
        obj2num[item] = count
        num2obj[count] = item

    def get_train_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Train graph contains " + str(len(self.train_data + self.inv_train_data)) + " triples\n")
        return np.array(self.train_data + self.inv_train_data, dtype=np.int64)

    def get_train_data(self):
        train_data = self.train_data + self.inv_train_data if self.include_reverse else self.train_data
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Train data contains " + str(len(train_data)) + " triples\n")
        return np.array(train_data, dtype=np.int64)

    def get_test_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Test graph contains " + str(len(self.train_data + self.inv_train_data)) + " triples\n")
        return np.array(self.train_data + self.inv_train_data, dtype=np.int64)

    def get_test_data(self):
        test_data = self.test_data + self.inv_test_data if self.include_reverse else self.test_data
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Test data contains " + str(len(test_data)) + " triples\n")
        return np.array(test_data, dtype=np.int64)

    def get_valid_data(self):
        valid_data = self.valid_data + self.inv_valid_data if self.include_reverse else self.valid_data
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Valid data contains " + str(len(valid_data)) + " triples\n")
        return np.array(valid_data, dtype=np.int64)