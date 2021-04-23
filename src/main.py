import os, sys
import argparse
import time
import torch
import logging as log
import numpy as np
from Data import Data_loader
from Trainer import Trainer

from Agent import Agent
from MetaAgent import MetaAgent

sys.path.extend(['../../coke/CoKE/bin'])

class Option:
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open (os.path.join(self.this_expsdir, "option.txt"), "w", encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key = lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))

    @classmethod
    def load(cls, path):
        args = []
        print(path)
        with open(os.path.join(path, "option.txt"), "r", encoding='UTF-8') as f:
            for line in f.readlines():
                print(line.strip())
                key, value = line.split(", ", 1)
                if key not in ["action_embed_size", "random_agent", "tag", "this_expsdir", "use_cuda"]:
                    args.extend([f"--{key.strip()}", value.strip().replace("False", "")])
        print(args)
        return cls.read_options(args)

    @classmethod
    def read_options(cls, args=None):
        parser = argparse.ArgumentParser(description="Experiment setup")

        # Log configuration
        parser.add_argument('--exps_dir', default="../exps/", type=str)
        parser.add_argument('--exp_name', default="demo", type=str)

        # Dataset
        parser.add_argument('--datadir', default="../datasets", type=str)
        parser.add_argument('--dataset', default="WN18_RR", type=str,
                            choices=["freebase15k_237", "WN18_RR"])
        parser.add_argument('--use_inverse', default=False, type=bool,
                            help='Set true to include inversed triples to the training.')

        # Agent configuration
        parser.add_argument('--mode', default='lstm_mlp', type=str,
                            choices=["lstm_mlp", "bert_mlp", "random", "coke_mlp", "coke_random"],
                            help='Which model to use: "lstm_mlp", "bert_mlp" or "random"')
        parser.add_argument("--meta", default=False, type=bool, help="")

        parser.add_argument('--state_embed_size', default=200, type=int,
                            help='Size of the context encoding (LSTM state or BERT reduced state representation)')
        parser.add_argument('--relation_embed_size', default=100, type=int, help='Size of the relation embeddings.')
        parser.add_argument('--mlp_hidden_size', default=200, type=int,  help='Size of the hidden MLP of the Agent.')
        parser.add_argument('--use_entity_embed', default=False, type=bool, help="")
        # parser.add_argument('--entity_embed_size', default=5, type=int)

        parser.add_argument('--train_times', default=20, type=int, help='Number of rollouts of the same episode (triple).')
        parser.add_argument('--test_times', default=100, type=int, help='Beam search size for one episode (triple) ')
        parser.add_argument("--train_batch", default=128, type=int,help='Number of training iterations.')
        parser.add_argument('--max_out', default=200, type=int, help='Maximal number of actions stored for one state.')
        parser.add_argument('--max_step_length', default=3, type=int)

        # Reward configuration
        parser.add_argument('--reward', default='context', type=str, help='Target to learn: "context" or "answer"')
        parser.add_argument('--metric', default='context', type=str,
                            help='How to evaluate the learned paths: "context" or "answer"')
        parser.add_argument('--bert_path', default='', type=str, help='Path to directory where the bert model is stored.')
        parser.add_argument('--load_config', default=False)
        parser.add_argument('--baseline', default='react', type=str)

        # Learning configuration
        parser.add_argument('--load_model', default='', type=str,
                            help='Path to the directory with the model file "model.pkt"')
        parser.add_argument('--load_option', default="", type=str)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=128, type=int, help='Batch size used for training.')
        parser.add_argument('--test_batch_size', default=64, type=int, help='Batch size used during evaluation.')

        # Decay of the beta: the entropy regularization factor
        parser.add_argument('--decay_weight', default=0.02, type=float)
        parser.add_argument('--decay_batch', default=200, type=int)
        parser.add_argument('--decay_rate', default=0.9, type=float)

        parser.add_argument('--gamma', default=1, type=float)
        parser.add_argument('--Lambda', default=0.02, type=float)
        parser.add_argument('--beta', default=0.02, type=float)

        parser.add_argument('--droprate', default=0.5, type=float)
        parser.add_argument('--token_droprate', default=0.4, type=float)

        parser.add_argument("--grad_clip_norm", default=5, type=int)
        parser.add_argument("--eval_batch", default=10, type=int,
                            help="How ofter to validate the model for mrr")

        # Randomization control
        parser.add_argument('--random_seed', default=2020, type=int)

        # Modi
        parser.add_argument('--do_test', default=False, type=bool,
                            help='If False, performs a short evaluation on small slices of train, dev and test data;\
                                  If True, performs a full evaluation on dev and test data.')

        # Trainable Bert
        parser.add_argument('--train_layers', default="", type=str)
        parser.add_argument('--bert_rate', default=10e-8, type=float)
        parser.add_argument('--bert_lr', default=10e-8, type=float)
        parser.add_argument('--bert_state_mode', default="avg_all", type=str, help='["avg_all", "avg_token", "sep"]')

        # coke
        parser.add_argument('--coke_len', default=7, type=int)
        parser.add_argument('--coke_mode', default="pqa", type=str) # pqa, anchor, lp
        parser.add_argument('--coke_config', type=str)
        parser.add_argument('--coke_model', type=str)
        parser.add_argument('--mask_head', default=False, type=bool)
        parser.add_argument('--epsilon', default=0.0, type=float)

        if args is None:
            d = vars(parser.parse_args())
        else:
            d = vars(parser.parse_args(args))
        print(d['use_entity_embed'])
        option = cls(d)
        print(option.use_entity_embed)
        return option

    def finalize_option(self):
        if self.load_model:
            self.load_option = self.load_model
        if self.exp_name is None:
            self.tag = time.strftime("%y-%m-%d-%H-%M")
        else:
            self.tag = self.exp_name

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.this_expsdir = os.path.join(self.exps_dir, self.tag)
        if not os.path.exists(self.exps_dir):
            os.makedirs(self.exps_dir)
        if not os.path.exists(self.this_expsdir):
            os.makedirs(self.this_expsdir)

        if not self.bert_path:
            if self.dataset == "WN18_RR":
                self.bert_path = "../../mastersthesis/transformers/knowledge_graphs/minerva_wn_a1/"
            elif self.dataset == "freebase15k_237":
                self.bert_path = "../../mastersthesis/transformers/knowledge_graphs/output_minevra_a1/"

        if "random" in self.mode:
            self.random = True
            self.test_times = 1
            self.train_batch = 0
        else:
            self.random = False

        if self.use_entity_embed is False:
            self.action_embed_size = self.relation_embed_size
        else:
            self.action_embed_size = self.relation_embed_size * 2  ## todo: allow different sizes via separate vocab
        
        self.train_layers = [int(l) for l in self.train_layers if l in "0123456789"]

def main(option):
    log.basicConfig(level=log.INFO)

    data_loader = Data_loader(option)
    option.num_entity = data_loader.num_entity
    option.num_relation = data_loader.num_relation
    agent = MetaAgent(option, data_loader) if option.meta else Agent(option, data_loader)
    trainer = Trainer(option, agent, data_loader)

    if option.load_model:
        trainer.load_model(name='last', exp_name=option.load_model)
        option.load_model = ''
    if option.train_batch != 0:
        trainer.train()
        trainer.save_model('last')
    if option.do_test:
        print("Eval last model")
        trainer.test(data='valid')
        trainer.test(data='test')
        print("Eval best model")
        if not option.random:
            trainer.load_model(name='best', exp_name=option.exp_name if option.train_batch != 0 else option.load_model)
            trainer.test(data='valid')
            trainer.test(data='test')
    else:
        #trainer.load_model()
        trainer.test(data='train', short=100)
        trainer.test(data='valid', short=100)
        trainer.test(data='test', short=100)

if __name__ == "__main__":
    #torch.set_printoptions(threshold=100000)
    option = Option.read_options()
    option.finalize_option()
    print(option.load_model, option.load_option, option.exp_name)
    print(f"Load option? {option.load_option}") 
    if option.load_option != "":
        new_option = Option.load(os.path.join(option.exps_dir, option.load_option))
        new_option.exp_name, new_option.load_model = option.exp_name, option.load_model
        new_option.meta = option.meta
        new_option.train_batch, new_option.batch_size, new_option.test_batch_size = option.train_batch, option.batch_size, option.test_batch_size
        new_option.reward, new_option.metric, new_option.bert_path = option.reward, option.metric, option.bert_path
        new_option.coke_len, new_option.mask_head = option.coke_len, option.mask_head
        new_option.coke_config, new_option.coke_model = option.coke_config, option.coke_model 
        new_option.mode = option.mode
        new_option.train_times = option.train_times
        new_option.finalize_option()
        option = new_option
    option.save()
    print(option.__dict__)

    main(option)



