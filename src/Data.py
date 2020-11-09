import os
from collections import defaultdict
import numpy as np

class Vocab:
    def __init__(self, base_entities, test_entities, relations):
        self.pad_token_id = 0

        # transformer tokens (CoKE)
        self.sep_token_id = 1
        self.mask_token_id = 2
        self.unk_token_id = 3
        self.cls_token_id = 4

        # RL agent tokens (Minerva)
        self.no_operation_token_id = 5  # MINERVA: no_op token
        self.start_token_id = 6

        self.reserved_vocab = 7
        self.next_idx, self.last_orig_idx, self.last_base_idx = self.reserved_vocab, None, None

        relation2num, entity2num = self.build_vocab(base_entities, test_entities, relations)

        self.item2num = defaultdict(lambda: self.unk_token_id)

        self.num2item = {
            self.pad_token_id: "PAD",
            self.unk_token_id: "UNK",
            self.no_operation_token_id: "NO_OP"
        }

        self.item2num.update(relation2num); self.item2num.update(entity2num)
        self.num2item.update({v: k for (k,v) in self.item2num.items()})

        self.num_entity = len(entity2num)
        self.num_relation = len(relation2num)

        # print data stats without reserved tokens
        print("#total relations", self.num_relation)
        print("#total entities", self.num_entity)

        # add reserved vocab to num_entities to have the last ent idx
        self.num_entity += self.reserved_vocab

        # get inv<->orig relation idx mapping
        self.rel2inv = {rel_idx: relation2num[self.get_inverse(rel)]
                        for rel, rel_idx in relation2num.items()}
        assert len(relation2num) == len(self.rel2inv)
        assert all(([self.get_inverse(self.num2item[rel_idx]) == self.num2item[inv_idx]
                 for rel_idx, inv_idx in self.rel2inv.items()]))
        self.rel2inv[self.cls_token_id] = self.cls_token_id

    def build_vocab(self, base_entities, test_entities, relations):
        """
        Derive the vocab ids by alphabetically sorting the elements.
        Order: relations, inverse relations, base entities (later emerging entities will be added)
        :param base_entities: list, entities occuring in the train/dev split
        :param test_entities: list, entities occuring in the test split
        :param relations: list, relations from the train split
        :return: None
        """
        base_entities = set(base_entities)
        test_entities = set(test_entities) - base_entities
        relations = set(relations)
        inv_relations = {self.get_inverse(rel) for rel in relations}

        print("#inv/orig relations", len(relations), len(inv_relations))
        print("#base/test entities", len(base_entities), len(test_entities))

        entity2num, relation2num = {}, {}

        # remember the last idx of base entities
        # to be able to tell base from test entities apart
        self.add_items(entity2num, base_entities)
        self.last_base_idx = self.next_idx -1
        self.add_items(entity2num, test_entities)

        # remember the last idx of original relations
        # to be able to tell original from inverse relation apart by index
        self.add_items(relation2num, relations)
        self.last_orig_idx = self.next_idx - 1
        self.add_items(relation2num, inv_relations)

        ## finilize for different relations and entity embedding sizes
        #     # map joint Transformer indices for relations and entities to separate mappings
        #     self.mixed2ent = {v: i for i, v in enumerate(self.kg.entity2idx.values())}
        #     self.ent2mixed = {v: k for k, v in self.mixed2ent.items()}
        #     # include reserved relations
        #     self.mixed2rel = {v: i for i, v in enumerate(self.kg.entity2idx.values())}
        #     self.rel2mixed = {v: k for k, v in self.mixed2rel.items()}
        #     # add pad token mapping
        #     self._add_item(self.mixed2rel, self.rel2mixed, 0)

        return relation2num, entity2num

    def add_items(self, mapping, items):
        mapping.update({
            item: self.next_idx + idx for idx, item in enumerate(sorted(items))
        })
        self.next_idx += len(items)

    def get_inverse(self, rel):
        return rel[2:] if rel.startswith('**') else '**' + rel

    def is_inverse_relation(self, np_array):
        if (np_array >= self.next_idx).any() \
                or (np_array < self.num_entity).any():
            raise TypeError("Passed relation array contains non-relational indices.")
        return np_array > self.last_orig_idx

    def is_test_entity(self, np_array, strict=True):
        if strict and ((np_array > self.num_entity).any() or (np_array < self.reserved_vocab).any()):
            print(np_array)
            raise TypeError("Passed entity array contains invalid indices.")
        return np_array > self.last_base_idx

    def dump(self, path):
        with open(path, "w") as fp:
            # order as in init
            for idx, token in enumerate(["PAD", "SEP", "MASK", "UNK", "CLS", "NO_OP", "START"]):
                fp.write(f"{token}\t{idx}\n")
            # note: num entity is already shifted by reserved vocab
            for idx in range(self.reserved_vocab, self.num_entity+self.num_relation):
                fp.write(f"{self.num2item[idx]}\t{idx}\n")


class DataLoader:
    def __init__(self, option):
        self.option = option
        self.include_reverse = option.use_inverse

        root_path = os.path.join(self.option.datadir, self.option.dataset)
        self.data_paths = {name: os.path.join(root_path, f"{name}.triples")
                           for name in ["train", "test", "aux"]}
        self.data_paths["valid"] = os.path.join(root_path, "dev.triples")

        raw_triple_by_split = self.read_raw_data()

        # manage vocab
        self.data_paths["vocab"] = os.path.join(root_path, "vocab.tsv")
        self.vocab = Vocab(*self.get_vocab_sets(raw_triple_by_split))
        self.write_vocab()

        # preprocess data by vocab
        self.data = {}
        self.prepare_data(raw_triple_by_split)

    def write_vocab(self):
        if os.path.exists(self.data_paths["vocab"]):
            print("Vocabulary file already exists, skip writing...")
            return
        self.vocab.dump(self.data_paths["vocab"])

    ### CONSTRUCT THE DATA READER ###
    def read_raw_data(self):
        '''
            read triplets from tsv files in dataset folder (in string format)
            :return
            Dict {split_name: nested list} with raw triples in s,p,o form
        '''
        triples = {}
        for data_split, path in self.data_paths.items():
            data = []
            with open(path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    s, o, p = line.strip().split('\t')
                    data.append([s.strip(), p.strip(), o.strip()])
            triples[data_split] = data
        return triples

    def get_vocab_sets(self, raw_triple_by_split):
        """
        Get dataset-specific splits of entities, relevant for induction setting.
        Gather entities and relations from different splits:
            base = {train, valid}
            test = {test, aux} ; note: test is a subset of aux
        :param raw_triple_by_split: {split_name: nested list} with raw triples in s,p,o form
        :return: sets of base, emerging (test) entities and of relations (as strings)
        """
        base_ents, test_ents, rels = set(), set(), set()
        for data_split, data in raw_triple_by_split.items():
            if data_split == "test":
                continue

            for triple in data:
                # collect all relations together
                rels.add(triple[1])

                # split entities by base/emerging
                so = {triple[0], triple[2]}
                if data_split in ["train", "valid"]:
                    base_ents.update(so)
                elif data_split == "aux":
                    test_ents.update(so)

        return base_ents, test_ents, rels

    def prepare_data(self, raw_triples_by_split):
        """
        Encode the data using the vocab, add inverted triples
        and augment original and inverted triples with inverse relation ids.
        Inverted triples can be used for bidirectional graph construction
        as well as for training with inverted queries.
        :param raw_triples_by_split: {split_name: nested list} with raw triples in s,p,o form
        :return: None
        """

        for data_split, data in raw_triples_by_split.items():
            self.data[data_split], self.data[f'inv_{data_split}']  = (
                [self._prepare_triple(triple,inv=False) for triple in data],
                [self._prepare_triple(triple, inv=True) for triple in data])

            assert not self.vocab.is_inverse_relation(np.array(self.data[data_split])[:, 1]).any()
            assert self.vocab.is_inverse_relation(np.array(self.data[f'inv_{data_split}'])[:, 1]).all()

            if data_split in ["train", "valid"]:
                assert not self.vocab.is_test_entity(np.array(self.data[data_split])[:, 0]).any()
                assert not self.vocab.is_test_entity(np.array(self.data[data_split])[:, 2]).any()
            elif data_split == "test":
                assert self.vocab.is_test_entity(np.array(self.data[data_split])[:, 0]).all()
                assert not self.vocab.is_test_entity(np.array(self.data[data_split])[:, 2]).any()
            else:
                # in aux triples there must be at least one base entity at any side
                assert not (
                    self.vocab.is_test_entity(np.array(self.data[data_split])[:, 0]) &
                    self.vocab.is_test_entity(np.array(self.data[data_split])[:, 2])
                ).any()

            print(f"{data_split}: {len(self.data[data_split])} facts")

    def _prepare_triple(self, triple, inv=False):
        h,r,t = [self.vocab.item2num[x] for x in triple]
        inv_r = self.vocab.rel2inv[r]
        return [t, inv_r, h, r] if inv else [h, r, t, inv_r]

    ### ACCESS THE DATA ###
    def get_data(self, data, include_inverse=True):
        out = self.data[data] + self.data[f'inv_{data}'] if include_inverse else self.data[data]
        return np.array(out, dtype=np.int64)

    def get_base_graph_data(self):
        return np.array(self.data['train'] + self.data['inv_train'], dtype=np.int64)

    def get_extended_graph_data(self):
        print(self.get_base_graph_data().shape)
        print(np.array(self.data['aux'] + self.data['inv_aux']).shape)
        return np.concatenate([
            self.get_base_graph_data(),
            np.array(self.data['aux'] + self.data['inv_aux'], dtype=np.int64)],
            axis=0
        )