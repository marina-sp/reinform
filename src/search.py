import numpy as np
import torch
import torch.optim as optim

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.factory import Models
from ax.service.ax_client import AxClient

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.suggest.ax import AxSearch
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch


def train(config):
    # transform dict to Namespace
        model.eval()
        with torch.no_grad():
            mrr = ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', reciprocal=args.reciprocal)
            # if epoch % 5 == 0:
            #     if epoch > 0:
            #         ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', reciprocal=args.reciprocal)

        tune.track.log(mean_accuracy=mrr)

ray.init(num_gpus=8, memory=15*1024*1024*1024)

np.random.seed(17)

# gen = GenerationStrategy(name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_arms=-1)])
#
# client = AxClient(random_seed=17, generation_strategy=gen, enforce_sequential_optimization=False)
# client.create_experiment(
#     parameters=search_space,
#     # parameter_constraints=['norm_l_ent >= 1 or reg_l_ent >= 1',
#     #                        'norm_l_rel >= 1 or reg_l_rel >= 1'],
#     objective_name='mean_accuracy',
#     minimize=False,  # Optional, defaults to False.
# )
#
# hyper_search = AxSearch(client, max_concurrent=4)

search_space = {
    "lr": hp.choice("lr", [0.01, 0.001, 0.0001, 0.00001]),
    "lr_decay": hp.choice("decay", [1, 0.995]),
    "label_smoothing": hp.choice("label_smoothing", [0, 0.1]),
    "embedding_dim": hp.choice("embedding_dim", [32, 64, 128, 256]),
    "activation": hp.choice("activation", ['tanh', 'linear']),
    "loss": loss,
    "loader_threads": 4,
    "opt": "adam",
    "lookahead": hp.choice("la", [True, False]),
    "reciprocal": hp.choice("reciprocal", [True, False]),
    "model": "transe",
    "score_norm": hp.choice("scoring", [1,2]),
    "type": hp.choice('type', ['KvsAll', '1vsAll']),
    "data": "FB15k-237",
    "batch_size": hp.choice('batch_size', [32, 128, 256]), "test_batch_size": 128,
    "epochs": 200,
    "conv_translation": False,

    'ent_penalty': hp.choice('ent_penalty', [
        {
            "norm_l_ent": hp.choice("norm_l_ent", [0,1,2]),
            "reg_l_ent":0,
        },
        {
            "reg_f_ent": hp.choice("reg_f_ent", [0.1**pow for pow in [5, 7, 9, 11, 13]]),
            "reg_l_ent": hp.choice("reg_l_ent", [1, 2, 3]),
            "reg_value_ent": hp.choice("reg_value_ent", [0, 1]),
            "norm_l_ent": 0
        }
    ]),

    'rel_penalty': hp.choice('rel_penalty', [
        {
            "norm_l_rel": hp.choice("norm_l_rel", [0, 1, 2]),
            "reg_l_rel": 0
        },
        {
            "reg_f_rel": hp.choice("reg_f_rel", [0.1 ** pow for pow in [5, 7, 9, 11, 13]]),
            "reg_l_rel": hp.choice("reg_l_rel", [1, 2, 3]),
            "reg_value_rel": hp.choice("reg_value_rel", [0, 1]),
            "norm_l_rel": 0
        }
    ])
}

hyper_search = HyperOptSearch(
    search_space, max_concurrent=12,
    metric="mean_accuracy", mode="max",
    random_state_seed=1995)

# scheduler = AsyncHyperBandScheduler(
#     time_attr='time_total_s',
#     metric='mean_accuracy',
#     mode="max",
#     max_t=12*60*60,
# )

scheduler = MedianStoppingRule(metric='mean_accuracy', mode="max")

class Stopper:
    def __init__(self):
        self.should_stop = False

    def stop(self, trial_id, result):
        t = result['training_iteration']
        q = result['mean_accuracy']
        def off_track(epoch, metric):
            if epoch >= 100:
                return metric <= 0.29
            elif epoch >= 40:
                return metric <= 0.25#7
            elif epoch >= 30:
                return metric <= 0.23#5
            elif epoch >= 20:
                return metric <= 0.19#20
            elif epoch >= 10:
                return metric <= 0.12#5
            elif epoch >= 5:
                return metric <= 0.09#10
            elif epoch >= 1:
                return metric <= 0.01
            else:
                return False

        if not self.should_stop and off_track(t, q):
            self.should_stop = True
        return self.should_stop

stopper = Stopper()

analysis = tune.run(
    train,
    name=name,
    search_alg=hyper_search,
    scheduler=scheduler,
    num_samples=N,
    resources_per_trial={"gpu": 1},
    verbose=1,
    #stop=stopper.stop
)

print(analysis.get_best_config(metric="mean_accuracy"))
print(analysis.trials)