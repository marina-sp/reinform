import numpy as np
import torch
import torch.optim as optim
import argparse

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.suggest.ax import AxSearch

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import run, sample_from
from ray.tune.schedulers import HyperBandScheduler

from Trainer import Trainer
from hyperopt import hp

np.random.seed(17)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()
if args.smoke_test:
    ray.init(num_cpus=4)
else:
    ray.init(num_gpus=4) #, memory=15*1024*1024*1024


# Hyperband early stopping, configured with `episode_reward_mean` as the
# objective and `training_iteration` as the time unit,
# which is automatically filled by Tune.
hyperband = HyperBandScheduler(
    time_attr="training_iteration",
    metric="hits@top",
    mode="max",
    max_t=1000)

space = {
    # Dataset
    'datadir': "../datasets",
    'dataset': "WN18_RR",   #["freebase15k_237", "WN18_RR"])
    'bert_path': "../../mastersthesis/transformers/knowledge_graphs/minerva_output_a1", #'Path to directory where the bert model is stored.')
    'use_inverse': False,   # 'Set true to include inversed triples to the training.')

    # Agent configuration
    'agent_mode': hp.choice('agent_mode', [
        {
            'mode': "lstm_mlp"
        },
        {
            'mode': "bert_mlp",
            'bert_state_mode': hp.choice('state_mode', ["avg_all", "avg_token", "sep"])
        }
    ]),  # 'Which model to use: "lstm_mlp", "bert_mlp" or "random"')
    'state_embed_size': hp.choice('state_size', [100, 200]),    # 'Size of the context encoding (LSTM state or BERT reduced state representation)')
    'relation_embed_size': hp.choice('rel_emb_size', [50,100,200]),   #'Size of the relation embeddings.')
    'mlp_hidden_size': hp.choice('mlp_size', [100,200,300]),   #'Size of the hidden MLP of the Agent.')
    'use_entity_embed': hp.choice('ent_emb', [True, False]),
    # 'entity_embed_size', default=5, type=int)

    'train_times': hp.choice('train_unrolls', [1,5,10,20]),   #'Number of rollouts of the same episode (triple).')
    #'test_times', default=100,   #'Beam search size for one episode (triple) ')
    "train_batch": 1000,   #'Number of training iterations.')
    'max_out': 200,   #'Maximal number of actions stored for one state.')
    'max_step_length': 3,

    # Reward configuration
    'reward': 'context',   #'Target to learn: "context" or "answer"')
    'metric': 'context',  #'How to evaluate the learned paths: "context" or "answer"')
    #'load_config', default=False)
    'baseline': 'react',

    # Learning configuration
    'learning_rate': hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001]),
    'batch_size': 128,   #'Batch size used for training.')
    #'test_batch_size', default=64,   #'Batch size used during evaluation.')

    # Decay of the beta: the entropy regularization factor
    'decay_weight':0.02,
    'decay_batch':200,
    'decay_rate':0.9,
    'gamma':1,
    'Lambda':0.02,
    'beta':0.02,
    "grad_clip_norm": 5,

    # Randomization control
    'random_seed': 1,

    # Trainable Bert
    'bert_training': hp.choice('bert_training', [
        {
            'train_layers': [],
            'bert_rate': 1,
            'bert_lr': 0,
            'bert_state_mode': "sep"
        },
        {
            'train_layers': hp.choice('train_layers', [
                [5], [5,4], [5,4,3], [5,4,3,2], [5,4,3,2,1], [5,4,3,2,1,0]
            ]),
            'bert_rate': hp.choice('bert_loss_part', [10e-3, 10e-5, 10e-8, 10e-10]),
            'bert_lr': hp.choice('bert_lr', [10e-3, 10e-5, 10e-8, 10e-10]),
        }
    ])
}

current_best_params = [
    {
        # Dataset
        'datadir': "../datasets",
        'dataset': "WN18_RR",   #["freebase15k_237", "WN18_RR"])
        'bert_path': "../../mastersthesis/transformers/knowledge_graphs/minerva_output_a1", #'Path to directory where the bert model is stored.')
        'use_inverse': False,   # 'Set true to include inversed triples to the training.')

        # Agent configuration
        'agent_mode': {
            'mode': "lstm_mlp"
            },  # 'Which model to use: "lstm_mlp", "bert_mlp" or "random"')
        'state_embed_size': 200,    # 'Size of the context encoding (LSTM state or BERT reduced state representation)')
        'relation_embed_size': 100,   #'Size of the relation embeddings.')
        'mlp_hidden_size': 200,   #'Size of the hidden MLP of the Agent.')
        'use_entity_embed': False,
        # 'entity_embed_size', default=5, type=int)

        'train_times': 20,   #'Number of rollouts of the same episode (triple).')
        #'test_times', default=100,   #'Beam search size for one episode (triple) ')
        "train_batch": 500,   #'Number of training iterations.')
        'max_out': 200,   #'Maximal number of actions stored for one state.')
        'max_step_length': 3,

        # Reward configuration
        'reward': 'context',   #'Target to learn: "context" or "answer"')
        'metric': 'context',  #'How to evaluate the learned paths: "context" or "answer"')
        #'load_config', default=False)
        'baseline': 'react',

        # Learning configuration
        'learning_rate': 0.001,
        'batch_size': 64,   #'Batch size used for training.')
        #'test_batch_size', default=64,   #'Batch size used during evaluation.')

        # Decay of the beta: the entropy regularization factor
        'decay_weight':0.02,
        'decay_batch':200,
        'decay_rate':0.9,
        'gamma':1,
        'Lambda':0.02,
        'beta':0.02,
        "grad_clip_norm": 5,

        # Randomization control
        'random_seed': 1,

        # Trainable Bert
        'bert_training':
            {
                'train_layers':
                    [5],
                'bert_rate': 1,
                'bert_lr': 10e-6,
            }
    },
    {
        # Dataset
        'datadir': "../datasets",
        'dataset': "WN18_RR",  # ["freebase15k_237", "WN18_RR"])
        'bert_path': "../../mastersthesis/transformers/knowledge_graphs/minerva_output_a1",
        # 'Path to directory where the bert model is stored.')
        'use_inverse': False,  # 'Set true to include inversed triples to the training.')

        # Agent configuration
        'agent_mode':
            {
                'mode': "bert_mlp",
                'bert_state_mode': "avg_all"
            },  # 'Which model to use: "lstm_mlp", "bert_mlp" or "random"')
        'state_embed_size': 200,    # 'Size of the context encoding (LSTM state or BERT reduced state representation)')
        'relation_embed_size': 100,   #'Size of the relation embeddings.')
        'mlp_hidden_size': 200,   #'Size of the hidden MLP of the Agent.')
        'use_entity_embed': False,
        # 'entity_embed_size', default=5, type=int)

        'train_times': 20,
        # 'Number of rollouts of the same episode (triple).')
        # 'test_times', default=100,   #'Beam search size for one episode (triple) ')
        "train_batch": 200,  # 'Number of training iterations.')
        'max_out': 200,  # 'Maximal number of actions stored for one state.')
        'max_step_length': 3,

        # Reward configuration
        'reward': 'context',  # 'Target to learn: "context" or "answer"')
        'metric': 'context',  # 'How to evaluate the learned paths: "context" or "answer"')
        # 'load_config', default=False)
        'baseline': 'react',

        # Learning configuration
        'learning_rate': 0.01,
        'batch_size': 64,  # 'Batch size used for training.')
        # 'test_batch_size', default=64,   #'Batch size used during evaluation.')

        # Decay of the beta: the entropy regularization factor
        'decay_weight': 0.02,
        'decay_batch': 200,
        'decay_rate': 0.9,
        'gamma': 1,
        'Lambda': 0.02,
        'beta': 0.02,
        "grad_clip_norm": 5,

        # Randomization control
        'random_seed': 1,

        # Trainable Bert
        'bert_training':
            {
                'train_layers': [5],
                'bert_rate': 10e-5,
                'bert_lr': 10e-8,
            }
    }
]

algo = HyperOptSearch(
    space, max_concurrent=1 if args.smoke_test else 4, metric="hits@top",
    #points_to_evaluate=current_best_params
)

analysis = run(
    Trainer,
    name="hyperband_test",
    num_samples=20,
    resources_per_trial={"gpu": 1},
    stop={"training_iteration": 1 if args.smoke_test else 99999},
    search_alg=algo,
    scheduler=hyperband,
    fail_fast=True)

print(analysis.get_best_config(metric="mean_accuracy"))
print(analysis.trials)