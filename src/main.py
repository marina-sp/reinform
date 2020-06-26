import os
import argparse
import time
import torch
import logging as log
import numpy as np
from Data import Data_loader
from Trainer import Trainer
from Agent import Agent

class Option:
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open (os.path.join(self.this_expsdir, "option.txt"), "w", encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key = lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))

def read_options():
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
                        choices=["lstm_mlp", "bert_mlp", "random"],
                        help='Which model to use: "lstm_mlp", "bert_mlp" or "random"')
    parser.add_argument('--state_embed_size', default=200, type=int,
                        help='Size of the context encoding (LSTM state or BERT reduced state representation)')
    parser.add_argument('--relation_embed_size', default=100, type=int, help='Size of the relation embeddings.')
    parser.add_argument('--mlp_hidden_size', default=200, type=int,  help='Size of the hidden MLP of the Agent.')
    parser.add_argument('--use_entity_embed', default=False, type=bool)
    # parser.add_argument('--entity_embed_size', default=5, type=int)

    parser.add_argument('--train_times', default=20, type=int, help='Number of rollouts of the same episode (triple).')
    parser.add_argument('--test_times', default=100, type=int, help='Beam search size for one episode (triple) ')
    parser.add_argument("--train_batch", default=128, type=int,help='Number of training iterations.')
    parser.add_argument('--max_out', default=200, type=int, help='Maximal number of actions stored for one state.')
    parser.add_argument('--max_step_length', default=3, type=int)

    # Reward configuration
    parser.add_argument('--reward', default='answer', type=str, help='Target to learn: "context" or "answer"')
    parser.add_argument('--metric', default='context', type=str,
                        help='How to evaluate the learned paths: "context" or "answer"')
    parser.add_argument('--bert_path', default='', type=str, help='Path to directory where the bert model is stored.')
    parser.add_argument('--load-config', default=False)
    parser.add_argument('--baseline', default='react', type=str)

    # Learning configuration
    parser.add_argument('--load_model', default='', type=str,
                        help='Path to the directory with the model file "model.pkt"')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size used for training.')
    parser.add_argument('--test_batch_size', default=5, type=int, help='Batch size used during evaluation.')

    # Decay of the beta: the entropy regularization factor
    parser.add_argument('--decay_weight', default=0.02, type=float)
    parser.add_argument('--decay_batch', default=200, type=int)
    parser.add_argument('--decay_rate', default=0.9, type=float)

    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--Lambda', default=0.02, type=float)
    parser.add_argument('--beta', default=0.02, type=float)

    parser.add_argument("--grad_clip_norm", default=5, type=int)

    # Randomization control
    parser.add_argument('--random_seed', default=1, type=int)

    # Modi
    parser.add_argument('--do_test', default=False, type=bool,
                        help='If False, performs a short evaluation on small slices of train, dev and test data;\
                              If True, performs a full evaluation on dev and test data.')

    d = vars(parser.parse_args())
    option = Option(d)

    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name

    if torch.cuda.is_available():
        option.use_cuda = True
    else:
        option.use_cuda = False

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.exps_dir):
        os.makedirs(option.exps_dir)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)

    if not option.bert_path:
        if option.dataset == "WN18_RR":
            option.bert_path = "../../mastersthesis/transformers/knowledge_graphs/wn_output_1000/"
        elif option.dataset == "freebase15k_237":
            option.bert_path = "../../mastersthesis/transformers/knowledge_graphs/output_minevra_a1/"

    if option.mode == "random":
        option.test_times = 1
        option.train_batch = 0

    if option.use_entity_embed is False:
        option.action_embed_size = option.relation_embed_size
    else:
        option.action_embed_size = option.relation_embed_size * 2  ## todo: allow different sizes via separate vocab


    option.save()
    print(option.__dict__)
    return option

def main(option):
    log.basicConfig(level=log.INFO)

    data_loader = Data_loader(option)
    option.num_entity = data_loader.num_entity
    option.num_relation = data_loader.num_relation
    agent = Agent(option, data_loader)
    trainer = Trainer(option, agent, data_loader)

    if option.load_model:
        trainer.load_model()
        option.load_model = ''
    if option.train_batch != 0:
        trainer.train()
        trainer.save_model()
    if option.do_test:
        #trainer.load_model()
        trainer.test(data='valid')
        trainer.test(data='test')
    else:
        #trainer.load_model()
        trainer.test(data='train', short=50)
        trainer.test(data='valid', short=50)
        trainer.test(data='test', short=50)

if __name__ == "__main__":
    #torch.set_printoptions(threshold=100000)
    option = read_options()
    main(option)



