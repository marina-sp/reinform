from pykg2vec.utils.kgcontroller import KnowledgeGraph
import random
import os
from numpy.random import choice
from collections import Counter
import numpy as np
from pykg2vec.utils.kgcontroller import Triple
import pandas as pd
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class KnowledgeGraphDataset(Dataset):
    def __init__(self, identifier, model, length, repeat=1, limit=None, dir="training_instances/"):
        if identifier not in ["MINEVRA"]:
            raise ValueError(f"{identifier} is not a valid identifier.")
        valid_models = ["A"]
        if model not in valid_models:
            raise ValueError(f"For mode {identifier}, {model} is not a valid model.")
        self.dir = dir
        self.identifier = identifier
        self.model = model
        self.length = length
        self.repeat = repeat
        self.limit = limit


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):

        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])

    def sanity_check(self):
        if self.identifier == "MINEVRA":
            if self.model == "A":
                return self._sanity_check(11,11)
        return False

    def _sanity_check(self, inputs_length, labels_length):
        i_conunter = Counter([len(x) for x in self.inputs])
        l_conunter = Counter([len(x) for x in self.labels])

        print("Inputs Counter:")
        print(i_conunter)

        print("Labels Counter:")
        print(l_conunter)

        if len(i_conunter)>1:
            print("Multiple counts for Input Counter")
            return False
        elif not inputs_length in i_conunter:
            print(f"Wrong input length, expected {str(inputs_length)}")
            return False
        if len(l_conunter)>1:
            print("Multiple counts for Input Counter")
            return False
        elif not labels_length in l_conunter:
            print(f"Wrong label length, expected {str(labels_length)}")
            return False
        return True


    def prepare_files(self, kg, data):
        file_prefix = f"{self.identifier}_{self.model}_{str(self.length)}_{str(self.repeat)}_{str(self.limit)}_"
        labels = []
        inputs = []
        file_labels = self.dir+file_prefix+data+"_labels"
        file_inputs = self.dir+file_prefix+data+"_inputs"
        if os.path.exists(file_labels) and os.path.exists(file_inputs):
            print("Loading features from cached file %s" % file_labels)
            with open(file_labels, 'rb') as handle:
                self.labels = pickle.load(handle)
            print("Loading features from cached file %s" % file_inputs)
            with open(file_inputs, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            triples = kg.triplets[data]
            if self.limit:
                triples = triples[0:self.limit]

            if self.identifier == "MINEVRA":
                if self.model == "A":
                    inputs, labels = kg.extract_path_rl_a(self.length,"RANDOM_WALK", triples, data, self.repeat)

            with open(file_labels,"wb") as handle:
                pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(file_inputs,"wb") as handle:
                pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.inputs = inputs
            self.labels = labels

        assert len(self.inputs) == len(self.labels)

class MinevraDataset(Dataset):
    def __init__(self, kg, file_path, mode):
        self.inputs = []
        self.labels = []
        df = pd.read_csv(file_path, delimiter="\t", names=["Triple","Path","Is_successful"])
        if mode == "SUCCESS":
            df = df[df["Is_successful"]==1]
        elif mode == "FAILURE":
            df = df[df["Is_successful"]==0]
        for _,row in df.iterrows()[0:10]:
            relation = kg.relation2idx[row["Triple"].split()[1]]
            relation = kg.get_rev_relation(relation)
            target = row["Triple"].split()[2]
            input = [kg.cls_token_id, kg.mask_token_id, relation]
            label = [-1, kg.entity2idx[target], -1]
            path = row["Path"]
            path = path.split()
            for e,element in enumerate(path):
                if e%2 == 0:
                    input.append(kg.entity2idx[element])
                else:
                    if element == "NO_OP":
                        input.append(kg.pad_token_id)
                    else:
                        input.append(kg.relation2idx[element])
                label.append(-1)
            
            self.inputs.append(input+[kg.sep_token_id])
            self.labels.append(label+[-1])


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):

        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])

class CustomKG(KnowledgeGraph):

    def __init__(self, path_length, dataset='freebase15k_237'):
        self.pad_token_id = 0
        self.sep_token_id = 1
        self.mask_token_id = 2
        self.unk_token_id = 3
        self.cls_token_id = 4

        self.path_length = path_length
        # Number of reserved vocabs
        self.reserved_vocab = 5
        super().__init__(dataset)

    def prepare_data(self):
        if self.dataset.cache_metadata_path.exists():
            os.remove(self.dataset.cache_metadata_path)
        super().prepare_data()

    def read_mappings(self):
        self.entity2idx = {v: k+self.reserved_vocab for k, v in enumerate(self.read_entities())}  ##
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        num_of_entities = len(list(self.entity2idx.keys()))
        self.relation2idx = {v: k+num_of_entities+self.reserved_vocab for k, v in enumerate(self.read_relations())}  ##
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

        self.vocab = {**self.entity2idx, **self.relation2idx}
        assert len(self.vocab) == len(self.entity2idx)+len(self.relation2idx)

    def add_reversed_relations(self):
        num_existing_relations = len(self.relations)
        self.num_orig_relations = num_existing_relations
        num_of_entities = len(list(self.entity2idx.keys()))
        # Add extra relations and their mapping
        rev_relations = ["_"+r for r in self.relations]
        self.relations = np.concatenate([self.relations,rev_relations])
        self.relation2idx = {v: k+num_of_entities+self.reserved_vocab for k, v in enumerate(self.relations)}  ##
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}
        self.vocab = {**self.entity2idx, **self.relation2idx}
        assert len(self.relations) == 2*num_existing_relations
        assert len(self.vocab) == len(self.entity2idx)+len(self.relation2idx)

        # Add reversed triples
        for data in ['train', 'valid', 'test']:
            temp_len = len(self.triplets[data])
            temp_data = []
            for triple in self.triplets[data]:
                rev_relation_id_1 = self.relation2idx["_"+self.idx2relation[triple.r]]
                rev_relation_id_2 = triple.r+num_existing_relations
                assert rev_relation_id_1==rev_relation_id_2
                temp_triple = Triple("","","")
                temp_triple.set_ids(triple.t,rev_relation_id_1,triple.h)

                # Update hr_t
                if (triple.t,rev_relation_id_1) not in self.hr_t:
                    self.hr_t[(triple.t,rev_relation_id_1)] = set()
                self.hr_t[(triple.t,rev_relation_id_1)].add(triple.h)
                if data in ["train","valid"]:
                    if data == "train":
                        current_hr_t = self.hr_t_train
                    else:
                        current_hr_t = self.hr_t_valid
                    if (triple.t,rev_relation_id_1) not in current_hr_t:
                        current_hr_t[(triple.t,rev_relation_id_1)] = set()
                    current_hr_t[(triple.t,rev_relation_id_1)].add(triple.h)

                # Update tr_h
                if (triple.h,rev_relation_id_1) not in self.tr_h:
                    self.tr_h[(triple.h,rev_relation_id_1)] = set()
                self.tr_h[(triple.h,rev_relation_id_1)].add(triple.t)
                if data in ["train","valid"]:
                    if data == "train":
                        current_tr_h = self.tr_h_train
                    else:
                        current_tr_h = self.tr_h_valid
                    if (triple.h,rev_relation_id_1) not in current_tr_h:
                        current_tr_h[(triple.h,rev_relation_id_1)] = set()
                    current_tr_h[(triple.h,rev_relation_id_1)].add(triple.t)

                # Add rev triple tp data
                temp_data.append(temp_triple)

            assert len(temp_data)==temp_len
            self.triplets[data].extend(temp_data)
    
    def add_extra_relations(self):
        idx = max(self.idx2relation.keys())+1
        self.relation2idx["NO_OP"] = idx
        self.idx2relation[idx] = "NO_OP"
        
        idx = max(self.idx2relation.keys())+1
        self.relation2idx["PAD"] = idx
        self.idx2relation[idx] = "PAD"

        idx = max(self.idx2relation.keys())+1
        self.relation2idx["START"] = idx
        self.idx2relation[idx] = "START"

        self.relations = np.append(self.relations, np.array(["NO_OP", "PAD", "START"])) 




    def read_2nd_order_relations(self):
        h_t_2 = {}
        for e,t in self.hr_t.items():
            for e1 in t:
                if e1 in self.h_t:
                    if e[0] in h_t_2:
                        h_t_2[e[0]].update(self.h_t[e1])
                    else:
                        h_t_2[e[0]] = self.h_t[e1].copy()
        self.h_t_2 = h_t_2

    def read_1st_order_relations(self):
        h_t = {}
        for e,t in self.hr_t.items():
            if e[0] in h_t:
                # temp = set(h_t[e[0]])
                # temp.add(list(t))
                h_t[e[0]].update(t)
            else:
                h_t[e[0]] = t.copy()
        self.h_t = h_t

    def prepare_interesting_entities(self, entities):
        self.interesting_entities = []
        for ent in entities:
            entity_inputs = []
            entity_labels = []
            data = self.hr_t_train
            for (h,r),t in data.items():
                if h == ent:
                    entity_inputs.extend([[self.cls_token_id,self.mask_token_id,r,x,self.sep_token_id] for x in t])
                    entity_labels.extend([[-1,ent,-1,-1,-1] for x in t])
                elif ent in t:
                    entity_inputs.append([self.cls_token_id,h,r,self.mask_token_id,self.sep_token_id])
                    entity_labels.append([-1,-1,-1,ent,-1])
            self.interesting_entities.append((entity_inputs, entity_labels))

    def prepare_interesting_relations(self, relations):
        self.interesting_relations = []
        for rel in relations:
            inputs = []
            labels = []
            data = self.hr_t_train
            for (h,r),t in data.items():
                if r == rel:
                    inputs.extend([[self.cls_token_id,h,r,x,self.sep_token_id] for x in t])
                    labels.extend([[-1,-1,r,-1,-1] for x in t])
            self.interesting_relations.append((inputs, labels))

    def merge_dicts(self):
        self.hr_t_train_valid = {**self.hr_t_train,**self.hr_t_valid}

    def __len__(self):
        return len(self.vocab)+self.reserved_vocab

    def extract_path_with_special_tokens(self, length, method="random", data="train", evaluate_all=False, p=10, q=10):
        # Account for CLS and SEP tokens
        if data == "train":
            return [self.cls_token_id] + self.extract_path_train(length, method, data, p=p, q=q) + [self.sep_token_id]
        else:
            if not evaluate_all:
                return self.extract_triple(data)
            else:
                return self.extract_all_triples(data)
        # return [self.cls_token_id] + path + [self.sep_token_id]

    def extract_path_train(self, length, method, data, p=10, q=10):
        if method == "BIASED_RANDOM_WALK":
            path = self.biased_random_walk(length, data, p=p, q=q)
        else:
            path = self.random_walk(length, data)

        # Apply padding if necessary
        if len(path)<length:
            pad_length = length-len(path)
            path = path + [self.pad_token_id]*pad_length

        assert len(path) == length
        return path
    def extract_path_eval(self, length, method, triple, data, mode="relation", p=10, q=10):
        if mode not in ["SOURCE", "RELATION", "TARGET", "SOURCE_ONLY", "TARGET_ONLY"]:
            raise ValueError('Not a valid validation mode: %s' % (mode))

        ## TODO: handle "all"
        if mode in ["SOURCE", "TARGET"]:
            temp_length = length-1
        else:
            temp_length = length/2

        random_h_r = (triple[0], triple[1])
        random_t = triple[2]
        path_1 = self.reversed_random_walk(temp_length, end_at=random_h_r[0], data=data)
        if method == "BIASED_RANDOM_WALK":
            path_2 = self.biased_random_walk(temp_length, start_at=random_t, p=p, q=q, data=data)
        else:
            path_2 = self.random_walk(temp_length, start_at=random_t, data=data)

        if mode == "SOURCE":
            path = [self.mask_token_id, random_h_r[1]] + path_2
            label = random_h_r[0]
        elif mode == "TARGET":
            path = path_1 + [random_h_r[1], self.mask_token_id]
            label = random_t
        else:
            path = path_1 + [self.mask_token_id] + path_2
            label = random_h_r[1]

        # Apply padding if necessary
        if len(path)<length:
            pad_length = length-len(path)
            path = path + [self.pad_token_id]*pad_length

        assert len(path) == length
        path = [self.cls_token_id] + path + [self.sep_token_id]
        labels = [-1]*len(path)
        label_idx = path.index(self.mask_token_id)
        labels[label_idx] = label
        return path, labels

    def extract_triple(self, split='valid'):
        # adj_dict = self.hr_t_valid
        # random_h_r = random.choice(list(adj_dict.keys()))
        # random_t = random.choice(list(adj_dict[random_h_r]))
        data = self.triplets[split]
        t = random.choice(data)
        return (t.h,t.r,t.t)

    def extract_all_triples(self, split='valid'):
        # adj_dict = self.hr_t_valid
        # res = []
        # for k,v in adj_dict:
        #     random_h_r = random.choice(list(adj_dict.keys()))
        #     random_t = random.choice(list(adj_dict[random_h_r]))
        #     res.append([random_h_r[0], random_h_r[1], random_t])
        data = self.triplets[split]
        res = [(t.h,t.r,t.t) for t in data]
        return res

    def extract_path_rl_a(self, length, method, triples, data, repeat=1, p=10, q=10):
        # if mode not in ["SOURCE"]:
        #     raise ValueError('Not a valid validation mode: %s' % (mode))

        length = length*2+3
        final_inputs, final_labels = [], []
        for triple in tqdm(triples, desc=f"Creating {data} data"):
            random_h_r = (triple.h, triple.r)
            random_t = triple.t
            for i in range(repeat):
                if method == "BIASED_RANDOM_WALK":
                    path_2 = self.biased_random_walk(length, start_at=random_t, p=p, q=q, data=data)
                else:
                    path_2 = self.random_walk(length, start_at=random_t, data=data, exclude_triple=triple)

                path = [self.mask_token_id, random_h_r[1]] + path_2
                label = random_h_r[0]

                # Apply padding if necessary
                if len(path)<length:
                    pad_length = length-len(path)
                    path = path + [self.pad_token_id]*pad_length

                assert len(path) == length
                path = [self.cls_token_id] + path + [self.sep_token_id]
                labels = [-1]*len(path)
                label_idx = path.index(self.mask_token_id)
                labels[label_idx] = label
                final_inputs.append(path)
                final_labels.append(labels)
        return final_inputs, final_labels


    def get_rev_relation(self, relation):
        if slef.idx2relation[relation] in ["PAD", "NO_OP", "START"]:
            return relation
        if relation >= self.reserved_vocab+len(self.entities)+self.num_orig_relations:
            rev_relation = relation-self.num_orig_relations
            assert self.idx2relation[relation]=="_"+self.idx2relation[rev_relation]
        else:
            rev_relation = relation+self.num_orig_relations
            assert "_"+self.idx2relation[relation]==self.idx2relation[rev_relation]

        return rev_relation

    def get_subgraph_starting_at(self, source, hr_list, exclude_t, exclude_r):
        res = []
        for k in hr_list:
            if k[0]==source and not (k[1]==exclude_r and source == exclude_t):
                res.append(k)
        return res

    def get_subgraph_ending_at(self, target, adj_dict):
        res = []
        for h_r, t in adj_dict.items():
            if target in list(t):
                res.append(h_r)
        return res

    def _get_biased_random_walk_probs(self, previous, current, options, p, q):
        probs = []
        for x in options:
            o = x[1]
            if o in self.h_t and previous in self.h_t[o]:
                probs.append(1)
            elif o == previous:
                probs.append(1/p)
            else:
                probs.append(1/q)
        assert len(probs) == len(options)
        normalized_probs = [x/sum(probs) for x in probs]
        return normalized_probs

    def random_walk(self, length, data="train", start_at=None, include_entities=True, exclude_triple=None):
        if data == "train":
            adj_dict = self.hr_t_train
        else:
            adj_dict = self.hr_t_train_valid

        path = []
        start_options = list(adj_dict.keys())
        exclude_r = self.get_rev_relation(exclude_triple.r)
        exclude_t = exclude_triple.t
        if start_at and self.get_subgraph_starting_at(start_at, list(adj_dict.keys()), exclude_t, exclude_r):
            start_options = self.get_subgraph_starting_at(start_at, list(adj_dict.keys()), exclude_t, exclude_r)
        elif start_at:
            return [start_at]
        random_h_r = random.choice(start_options)
        t_options = list(adj_dict[random_h_r])
        if random_h_r[1] == exclude_r:
            t_options.remove(exclude_triple.h)
            if not t_options:
                import ipdb; ipdb.set_trace()
        random_t = random.choice(t_options)
        if include_entities:
            path.extend([random_h_r[0], random_h_r[1], random_t])
        else:
            path.extend([random_h_r[0], random_h_r[1]])
        while len(path) < length-2:
            options = self.get_subgraph_starting_at(random_t, list(adj_dict.keys()), exclude_t, exclude_r)
            if options:
                random_h_r = random.choice(options)
                t_options = list(adj_dict[random_h_r])
                if random_h_r[0]==exclude_triple.t and random_h_r[1] == exclude_r:
                    try:
                        t_options.remove(exclude_triple.h)
                    except:
                        return path
                random_t = random.choice(t_options)
                if include_entities:
                    path.extend([random_h_r[1], random_t])
                else:
                    path.extend([random_h_r[1]])
            else:
                break
        if not include_entities:
            path = path + [random_t]

        return path

    def reversed_random_walk(self, length, data="train", end_at=None, include_entities=True):
        if data == "train":
            adj_dict = self.hr_t_train
        else:
            adj_dict = self.hr_t_train_valid

        path = []
        random_t = random.choice(list(self.idx2entity.keys()))
        if end_at and self.get_subgraph_ending_at(end_at, adj_dict):
            random_t = end_at
        elif end_at:
            return [end_at]
        start_options = self.get_subgraph_ending_at(random_t, adj_dict)
        random_h_r = random.choice(start_options)
        if include_entities:
            path.extend([random_h_r[0], random_h_r[1], random_t])
        else:
            path.extend([random_h_r[1], random_t])
        while len(path) < length-2:
            options = self.get_subgraph_ending_at(random_h_r[0], adj_dict)
            if options:
                random_h_r = random.choice(options)
                if include_entities:
                    path = [random_h_r[0], random_h_r[1]] + path
                else:
                    path = [random_h_r[1]] + path
            else:
                break
        if not include_entities:
            path = [random_h_r[0]] + path

        return path

    def biased_random_walk(self, length, data="train", start_at=None, p=100, q=0.5, include_entities=True):
        if data == "train":
            adj_dict = self.hr_t_train
        else:
            adj_dict = self.hr_t_train_valid
        path = []
        start_options = list(adj_dict.keys())
        if start_at and self.get_subgraph_starting_at(start_at, list(adj_dict.keys())):
            start_options = self.get_subgraph_starting_at(start_at, list(adj_dict.keys()))
        random_h_r = random.choice(start_options)
        random_t = random.choice(list(adj_dict[random_h_r]))
        if include_entities:
            path.extend([random_h_r[0], random_h_r[1], random_t])
        else:
            path.extend([random_h_r[0], random_h_r[1]])
        while len(path) < length-2:
            options = self.get_subgraph_starting_at(path[-1], list(adj_dict.keys()))
            options_r_t = set()
            for o in options:
                options_r_t.update([(o[1],x) for x in adj_dict[o]])
            probs = self._get_biased_random_walk_probs(path[-3], random_t, list(options_r_t), p, q)
            if options:
                random_r_t = random.choices(list(options_r_t), weights=probs)[0]
                random_t = random_r_t[1]
                if include_entities:
                    path.extend([random_r_t[0], random_t])
                else:
                    path.extend([random_r_t[0]])
            else:
                break
        return path

    def extract_path_bd(self, triples, length, data):
        inputs, labels = [], []

        for t in tqdm(triples, desc=f"Creating {data} data"):
            t = (t.h, t.r, t.t)
            t_path, t_label = self.extract_path_eval(length*2+3, "RANDOM_WALK", t, data, "TARGET")
            h_path, h_label = self.extract_path_eval(length*2+3, "RANDOM_WALK", t, data, "SOURCE")
            assert len(t_path) == len(t_label) == len(h_path) == len(h_label)
            inputs.append(t_path)
            inputs.append(h_path)
            labels.append(t_label)
            labels.append(h_label)
        return inputs, labels

    def extract_path_e(self, triples, length, data, repeat=2):
        inputs, labels = [], []

        for t in tqdm(triples, desc=f"Creating {data} data"):
            t = (t.h, t.r, t.t)
            for i in range(repeat):
                t_path, t_label = self.extract_path_eval(length*2+3, "RANDOM_WALK", t, data, "TARGET")
                h_path, h_label = self.extract_path_eval(length*2+3, "RANDOM_WALK", t, data, "SOURCE")
                assert len(t_path) == len(t_label) == len(h_path) == len(h_label)
                inputs.append(t_path)
                inputs.append(h_path)
                labels.append(t_label)
                labels.append(h_label)
        return inputs, labels

    def extract_path_a(self, triples, length, data):
        def _extract_triples_from_path(path, has_cls, add_sep, kg):
            triples = []
            if has_cls:
                path = path[1:]
            for i in range(1,len(path)-1,2):
                triples.extend(path[i-1:i+2])
                if add_sep:
                    triples.append(kg.sep_token_id)
            return triples

        inputs, labels = [], []
        max_length = (length+1)*4+1

        for t in tqdm(triples, desc=f"Creating {data} data"):
            t = (t.h, t.r, t.t)
            path = self.reversed_random_walk(length*2+3, end_at=t[0], include_entities=True, data=data)
            path = path + [t[1], self.mask_token_id]
            path = [self.cls_token_id] + _extract_triples_from_path(path, False, True, self)
            while(len(path)) < max_length:
                path += [self.pad_token_id, self.pad_token_id, self.pad_token_id, self.sep_token_id]
            inputs.append(path)
            l = [-1]*len(path)
            label_idx = path.index(self.mask_token_id)
            l[label_idx] = t[2]
            labels.append(l)

            path = self.random_walk(length*2+3, start_at=t[2], include_entities=True, data=data)
            path = [self.mask_token_id, t[1]] + path
            path = [self.cls_token_id] + _extract_triples_from_path(path, False, True, self)
            while(len(path)) < max_length:
                path += [self.pad_token_id, self.pad_token_id, self.pad_token_id, self.sep_token_id]
            # path = self.pad_and_prepare_path([self.mask_token_id, t[1], t[2], self.sep_token_id]+path, max_length)
            inputs.append(path)
            l = [-1]*len(path)
            label_idx = path.index(self.mask_token_id)
            l[label_idx] = t[0]
            labels.append(l)

        return inputs, labels

    def extract_path_c(self, triples, length, data):
        def _extract_triples_from_path(path, has_cls, add_sep, kg):
            triples = []
            if has_cls:
                path = path[1:-1]
            for i in range(1,len(path),2):
                triples.extend(path[i-1:i+2])
                if add_sep:
                    triples.append(kg.sep_token_id)
            return triples

        inputs, labels = [], []
        max_length = (length+1)*3

        for t in tqdm(triples, desc=f"Creating {data} data"):
            t = (t.h, t.r, t.t)
            path = self.reversed_random_walk(length*2+2, end_at=t[0], include_entities=True, data=data)
            path = _extract_triples_from_path(path, False, False, self)
            path = self.pad_and_prepare_path(path + [t[0], t[1], self.mask_token_id], max_length)
            inputs.append(path)
            l = [-1]*((len(path)-2)//3+2)
            label_idx = path.index(self.mask_token_id)
            l[label_idx//3] = t[2]
            labels.append(l)
            assert len(l) == (len(path)-2)//3+2

            path = self.random_walk(length*2+2, start_at=t[2], include_entities=True, data=data)
            path = _extract_triples_from_path(path, False, False, self)
            path = self.pad_and_prepare_path([self.mask_token_id, t[1], t[2]]+path, max_length)
            inputs.append(path)
            l = [-1]*((len(path)-2)//3+2)
            # label_idx = path.index(self.mask_token_id)
            l[1] = t[0]
            labels.append(l)

            assert len(l) == (len(path)-2)//3+2

        return inputs, labels
    def _get_neighborhood(self, triple, max_length, mask="h", data="train", add_sep=True, same_label_length=True):
        if max_length and add_sep:
            max_length = max_length*2
        if data == "train":
            adj_dict = self.hr_t_train
        else:
            adj_dict = self.hr_t_train_valid
        h,r,t = triple
        if mask == "h":
            subject = t
        else:
            subject = h

        input = [self.cls_token_id]
        count = 1
        starts = self.get_subgraph_starting_at(subject, adj_dict)
        ends = self.get_subgraph_ending_at(subject, adj_dict)
        options = starts+ends
        random.shuffle(options)
        for o in options:
            if max_length and count>max_length:
                break
            if o in starts:
                tail = random.choice(list(adj_dict[o]))
                if not (tail == t and subject == h):
                    if add_sep:
                        input.extend([subject,o[1],tail,self.sep_token_id])
                        count += 2
                    else:
                        input.extend([subject,o[1],tail])
                        count += 1
            else:
                if not (o[0] == h and subject == t):
                    if add_sep:
                        input.extend([o[0],o[1],subject,self.sep_token_id])
                        count += 2
                    else:
                        input.extend([o[0],o[1],subject])
                        count += 1

        if max_length:
            while count<=max_length:
                if add_sep:
                    input.extend([self.pad_token_id,self.pad_token_id,self.pad_token_id,self.sep_token_id])
                    count += 2
                else:
                    input.extend([self.pad_token_id,self.pad_token_id,self.pad_token_id])
                    count += 1

        # if not add_sep:
        #     input.extend([self.sep_token_id])
        #     count += 1

        # Use for subgraph + triples
        if not same_label_length:
            label = [-1]*count
            assert len(label) == (len(input)-2)//3+2
        else:
            # Use for subgraph + conv
            label = [-1]*len(input)
            assert len(label) == len(input)
        return input, label

    def extract_subgraph_a(self, triples, length, data):
        inputs, labels = [], []
        for t in tqdm(triples, desc=f"Creating {data} data"):
            t_tuple = (t.h, t.r, t.t)
            input, label = self._get_neighborhood(t_tuple, length, mask="h", data=data)
            last_triple = [self.mask_token_id,t.r,t.t,self.sep_token_id]
            input += last_triple
            label += [t.h,-1,-1,-1]
            inputs.append(input)
            labels.append(label)
            input, label = self._get_neighborhood(t_tuple, length, mask="t", data=data)
            last_triple = [t.h,t.r,self.mask_token_id,self.sep_token_id]
            input += last_triple
            label += [-1,-1,t.t,-1]
            inputs.append(input)
            labels.append(label)

        return inputs,labels

    def extract_subgraph_c(self, triples, length, data):
        inputs, labels = [], []
        for t in tqdm(triples, desc=f"Creating {data} data"):
            t_tuple = (t.h, t.r, t.t)
            input, label = self._get_neighborhood(t_tuple, length, mask="h", data=data, add_sep=False, same_label_length=False)
            last_triple = [self.mask_token_id,t.r,t.t,self.sep_token_id]
            input += last_triple
            label += [t.h,-1]
            inputs.append(input)
            labels.append(label)
            input, label = self._get_neighborhood(t_tuple, length, mask="t", data=data, add_sep=False, same_label_length=False)
            last_triple = [t.h,t.r,self.mask_token_id,self.sep_token_id]
            input += last_triple
            label += [t.t,-1]
            inputs.append(input)
            labels.append(label)

        return inputs,labels

    def pad_and_prepare_path(self, path, max_length, add_cls_sep=True):
        if len(path)<max_length:
            pad_length = max_length-len(path)
            path = path + [self.pad_token_id]*pad_length
        if add_cls_sep:
            if path[-1] == self.sep_token_id:
                path = [self.cls_token_id] + path
            else:
                path = [self.cls_token_id] + path + [self.sep_token_id]
        return path

    def get_special_tokens_mask(self, token_ids):
        return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id, self.pad_token_id] else 0, token_ids))
