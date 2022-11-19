Initially forked from [https://github.com/ty4b112/pytorch_MINERVA](https://github.com/ty4b112/pytorch_MINERVA) (a custom attempt to reimplement [1] with PyTorch)

# ReInform: Selecting paths with reinforcement learning for contextualized link prediction

This is the code for training the RL-based path search in the graph. A **RE**inforcement learning agent attempts to find the most **inform**ative path in the graph, starting from a query entity, such that a pretrained scoring sequence model (e.g. CoKE [2]) can best predict a missing entity from this path. In other words, RL agents perform context search to provide the most informative context for a given triple completion query. Sequence models that evaluate paths and do an actual prediction of the missing entity have to be pretrained separately.


## Quick call
```
python src/main.py
```

Sample call for model training (denoted *Transform-CoKE + RL* in the paper), which uses a pretrained CoKE-like model as the top predictor and trains an RL model to select a 2-step context.

```
python3 main.py --mode coke_mlp --max_step_length 2 --test_times 100 --do_test True --token_droprate 0 --relation_embed_size 50 --state_embed_size 100 --mlp_hidden_size 150 --droprate 0.4 --train_batch 1000 --batch_size 256 --train_times 60 --beta 0.05 --decay_batch 100 --exp_name wn_coke_test --eval_batch 5 --learning_rate 0.005 --use_inverse True --use_entity_embed True --coke_len "-1"
```

For a BERT-interent like architecture, where the sequence scoring model is a pretrained BERT, and the LSTM-based RL agent is trained to select a 2-step context:

```
python3 main.py --mode lstm_mlp --max_step_length 2 --test_times 100 --do_test True --relation_embed_size 100 --state_embed_size 200 --mlp_hidden_size 200 --train_batch 500 --batch_size 32 --test_batch_size 8 --train_times 20 --beta 0.05 --decay_batch 100 --exp_name wn_bert_lstm_final --learning_rate 0.001 --use_inverse True --use_entity_embed False --bert_path ../../mastersthesis/transformers/knowledge_graphs/wn_layer12_epoch100/
```
 
 
## Requirements
To use CoKE as the top sequence scorer, the [CoKE library](https://github.com/PaddlePaddle/Research/tree/master/KG/CoKE) should be placed locally ([s. module loading example](https://github.com/marina-sp/reinform/blob/e632f644b8e79bbfcc63de16bd5830bdeac06b5a/src/coke.py#L15)). The path to the loaded model should be set [here](https://github.com/marina-sp/reinform/blob/e632f644b8e79bbfcc63de16bd5830bdeac06b5a/src/coke.py#L171).

Alternatively, use a pretrained BERT-like model and provide the model path through the CL arguments or set as default [here](https://github.com/marina-sp/reinform/blob/master/src/main.py#L75). Make sure the vocabularies of the model and the RL agent match.


[1] [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)

[2] [CoKE: Contextualized Knowledge Graph Embedding](https://arxiv.org/abs/1911.02168)

