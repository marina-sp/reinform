from pykg2vec.utils.kgcontroller import KnowledgeGraph
import random
import os
from numpy.random import choice
from collections import Counter
import numpy as np
from pykg2vec.utils.kgcontroller import Triple

class CustomKG(KnowledgeGraph):

    def __init__(self, path_length, dataset='freebase15k_237'):
        self.pad_token_id = 0
        self.sep_token_id = 1
        self.mask_token_id = 2
        self.unk_token_id = 3
        self.cls_token_id = 4

        # Number of reserved vocabs
        self.reserved_vocab = 5
        super().__init__(dataset)

        self.rel2inv = {}

    def prepare_data(self):
        if self.dataset.cache_metadata_path.exists():
            os.remove(self.dataset.cache_metadata_path)
        super().prepare_data()

    def read_mappings(self):
        self.entity2idx = {v: k + self.reserved_vocab for k, v in enumerate(self.read_entities())}  ##
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        num_of_entities = len(list(self.entity2idx.keys()))
        self.relation2idx = {v: k + num_of_entities + self.reserved_vocab for k, v in
                             enumerate(self.read_relations())}
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

        self.vocab = {**self.entity2idx, **self.relation2idx}
        assert len(self.vocab) == len(self.entity2idx) + len(self.relation2idx)

    def add_reversed_relations(self):
        num_existing_relations = len(self.relations)
        self.num_orig_relations = num_existing_relations
        num_of_entities = len(list(self.entity2idx.keys()))
        # Add extra relations and their mapping
        rev_relations = ["_" + r for r in self.relations]
        self.relations = np.concatenate([self.relations, rev_relations])
        self.relation2idx = {v: k + num_of_entities + self.reserved_vocab for k, v in enumerate(self.relations)}  ##
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}
        self.vocab = {**self.entity2idx, **self.relation2idx}
        assert len(self.relations) == 2 * num_existing_relations
        assert len(self.vocab) == len(self.entity2idx) + len(self.relation2idx)

        # Add reversed triples
        for data in ['train', 'valid', 'test']:
            #self.triplets[data] = self.triplets[data][:100]
            temp_len = len(self.triplets[data])
            temp_data = []
            for triple in self.triplets[data]:
                rev_relation_id_1 = self.relation2idx["_" + self.idx2relation[triple.r]]
                rev_relation_id_2 = triple.r + num_existing_relations
                assert rev_relation_id_1 == rev_relation_id_2
                self.rel2inv[triple.r], self.rel2inv[rev_relation_id_1] = rev_relation_id_1, triple.r
                temp_triple = Triple("", "", "")
                temp_triple.set_ids(triple.t, rev_relation_id_1, triple.h)

                # Update hr_t
                if (triple.t, rev_relation_id_1) not in self.hr_t:
                    self.hr_t[(triple.t, rev_relation_id_1)] = set()
                self.hr_t[(triple.t, rev_relation_id_1)].add(triple.h)
                if data in ["train", "valid"]:
                    if data == "train":
                        current_hr_t = self.hr_t_train
                    else:
                        current_hr_t = self.hr_t_valid
                    if (triple.t, rev_relation_id_1) not in current_hr_t:
                        current_hr_t[(triple.t, rev_relation_id_1)] = set()
                    current_hr_t[(triple.t, rev_relation_id_1)].add(triple.h)

                # Update tr_h
                if (triple.h, rev_relation_id_1) not in self.tr_h:
                    self.tr_h[(triple.h, rev_relation_id_1)] = set()
                self.tr_h[(triple.h, rev_relation_id_1)].add(triple.t)
                if data in ["train", "valid"]:
                    if data == "train":
                        current_tr_h = self.tr_h_train
                    else:
                        current_tr_h = self.tr_h_valid
                    if (triple.h, rev_relation_id_1) not in current_tr_h:
                        current_tr_h[(triple.h, rev_relation_id_1)] = set()
                    current_tr_h[(triple.h, rev_relation_id_1)].add(triple.t)

                # Add rev triple tp data
                temp_data.append(temp_triple)

            assert len(temp_data) == temp_len
            self.triplets[data].extend(temp_data)

    def add_extra_relations(self):
        idx = max(self.idx2relation.keys()) + 1
        self.relation2idx["NO_OP"] = idx
        self.idx2relation[idx] = "NO_OP"

        idx = max(self.idx2relation.keys()) + 1
        self.relation2idx["PAD"] = idx
        self.idx2relation[idx] = "PAD"

        idx = max(self.idx2relation.keys()) + 1
        self.relation2idx["START"] = idx
        self.idx2relation[idx] = "START"

        self.relations = np.append(self.relations, np.array(["NO_OP", "PAD", "START"]))
