import argparse
import collections
import multiprocessing
import os
import time
import logging
import json
import random
import sys

import numpy as np
import paddle
import paddle.fluid as fluid
import torch

sys.path.extend(["../../coke/CoKE/bin"])

from reader.coke_reader import KBCDataReader
from model.coke import CoKEModel
from optimization import optimization
# from evaluation import kbc_evaluation
from evaluation import kbc_batch_evaluation
from evaluation import compute_kbc_metrics
from evaluation import pathquery_batch_evaluation
from evaluation import compute_pathquery_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

from run import init_coke_net_config, init_predict_checkpoint, init_checkpoint

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import paddle
import paddle.fluid as fluid

from reader.coke_reader import KBCDataReader
from model.coke import CoKEModel
from optimization import optimization
#from evaluation import kbc_evaluation
from evaluation import kbc_batch_evaluation
from evaluation import compute_kbc_metrics
from evaluation import pathquery_batch_evaluation
from evaluation import compute_pathquery_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

def create_model(pyreader_name, coke_config, args):
    pyreader = fluid.layers.py_reader\
            (
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
        dtypes=[
            'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    (src_ids, pos_ids, input_mask, mask_labels, mask_positions) = fluid.layers.read_file(pyreader)

    coke = CoKEModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        input_mask=input_mask,
        config=coke_config,
        soft_label=args.soft_label,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_fp16)

    loss, fc_out = coke.get_pretraining_output(mask_label=mask_labels, mask_pos=mask_positions)
    if args.use_fp16 and args.loss_scaling > 1.0:
        loss = loss * args.loss_scaling
    
    embs = coke.get_embeddings()

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=mask_labels, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    return embs, pyreader, loss, fc_out, num_seqs


## MODEL SETUP ##
class CoKEWrapper:
    def __init__(self, coke_mode, rel2inv_map, dataset='wn', max_len=-1, mask_head=False):
        super().__init__()

        self.rel2inv = rel2inv_map
        self.coke_mode = coke_mode

        # how sequences are formed on the input
        self.mask_head = mask_head

        if dataset.lower().startswith("w"):
            if max_len == -1:   
                config_name = "pathqueryWN18RR_config"
            elif max_len in [4,5,6]:
                config_name = f"pathqueryWN18RR_len{max_len - 2}_config"
            #config_name = "pathqueryWN18RR_lp_len3_128dim_config"
        else:
            config_name = "pathqueryFB237_config"
 
        config_file = f"../../coke/CoKE/configs/{config_name}.sh"
        # read CoKE config
        with open(config_file, "r") as f:
            content = f.read()
        params = [line.split("=") for line in content.split("\n") if line.strip() != "" and not line.startswith("#")]
        config = [part for param_name, param_value in params
                  for part in [f"--{param_name.strip().lower()}", param_value.strip()]
                      if param_name not in ["TASK", "NUM_VOCAB", "OUTPUT", "VALID_FILE", "TEST_FILE", "SEN_CANDLI_PATH",
                          "TRIVAL_SEN_PATH", "LOG_FILE", "LOG_EVAL_FILE", "MAX_POSITION_EMBEDDINS"]]
        print(config)
        config.extend(["--do_predict", "True"])

        parser = argparse.ArgumentParser()
        model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg("hidden_size",              int, 256,            "CoKE model config: hidden size, default 256")
        model_g.add_arg("num_hidden_layers",        int, 6,              "CoKE model config: num_hidden_layers, default 6")
        model_g.add_arg("num_attention_heads",      int, 4,              "CoKE model config: num_attention_heads, default 4")
        model_g.add_arg("vocab_size",               int, -1,           "CoKE model config: vocab_size")
        model_g.add_arg("num_relations",         int, None,           "CoKE model config: vocab_size")
        model_g.add_arg("max_position_embeddings",  int, 10,             "CoKE model config: max_position_embeddings")
        model_g.add_arg("hidden_act",               str, "gelu",         "CoKE model config: hidden_ac, default gelu")
        model_g.add_arg("hidden_dropout_prob",      float, 0.1,          "CoKE model config: attention_probs_dropout_prob, default 0.1")
        model_g.add_arg("attention_probs_dropout_prob", float, 0.1,      "CoKE model config: attention_probs_dropout_prob, default 0.1")
        model_g.add_arg("initializer_range",        int, 0.02,           "CoKE model config: initializer_range")
        model_g.add_arg("intermediate_size",        int, 512,            "CoKE model config: intermediate_size, default 512")

        model_g.add_arg("init_checkpoint",          str,  None,          "Init checkpoint to resume training from, or for prediction only")
        model_g.add_arg("init_pretraining_params",  str,  None,          "Init pre-training params which preforms fine-tuning from. If the "
                         "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
        model_g.add_arg("checkpoints",              str,  "checkpoints", "Path to save checkpoints.")
        model_g.add_arg("weight_sharing",           bool, True,          "If set, share weights between word embedding and masked lm.")

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch",             int,    100,                "Number of epoches for training.")
        train_g.add_arg("learning_rate",     float,  5e-5,               "Learning rate used to train with warmup.")
        train_g.add_arg("lr_scheduler",     str, "linear_warmup_decay",  "scheduler of learning rate.",
                        choices=['linear_warmup_decay', 'noam_decay'])
        train_g.add_arg("soft_label",               float, 0.9,          "Value of soft labels for loss computation")
        train_g.add_arg("weight_decay",      float,  0.01,               "Weight decay rate for L2 regularizer.")
        train_g.add_arg("warmup_proportion", float,  0.1,                "Proportion of training steps to perform linear learning rate warmup for.")
        train_g.add_arg("use_ema",           bool,   True,               "Whether to use ema.")
        train_g.add_arg("ema_decay",         float,  0.9999,             "Decay rate for expoential moving average.")
        train_g.add_arg("use_fp16",          bool,   False,              "Whether to use fp16 mixed precision training.")
        train_g.add_arg("loss_scaling",      float,  1.0,                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

        log_g = ArgumentGroup(parser, "logging", "logging related.")
        log_g.add_arg("skip_steps",          int,    1000,               "The steps interval to print loss.")
        log_g.add_arg("verbose",             bool,   False,              "Whether to output verbose log.")

        data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("dataset",                   str,   "",    "dataset name")
        data_g.add_arg("train_file",                str,   None,  "Data for training.")
        data_g.add_arg("sen_candli_file",           str,   None,  "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
        data_g.add_arg("sen_trivial_file",           str,   None,  "trivial sentence file for pathquery evaluation. Only used for path query datasets")
        data_g.add_arg("predict_file",              str,   None,  "Data for predictions.")
        data_g.add_arg("vocab_path",                str,   None,  "Path to vocabulary.")
        data_g.add_arg("true_triple_path",          str,   None,  "Path to all true triples. Only used for KBC evaluation.")
        data_g.add_arg("max_seq_len",               int,   3,     "Number of tokens of the longest sequence.")
        data_g.add_arg("batch_size",                int,   12,    "Total examples' number in batch for training. see also --in_tokens.")
        data_g.add_arg("in_tokens",                 bool,  False,
                       "If set, the batch size will be the maximum number of tokens in one batch. "
                       "Otherwise, it will be the maximum number of examples in one batch.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("do_train",                     bool,   False,  "Whether to perform training.")
        run_type_g.add_arg("do_predict",                   bool,   False,  "Whether to perform prediction.")
        run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training, default is True.")
        run_type_g.add_arg("use_fast_executor",            bool,   False,  "If set, use fast parallel executor (in experiment).")
        run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,      "Ihe iteration intervals to clean up temporary variables.")
        
        print("argparser set")
        self.args = parser.parse_args(config)
        self.args.do_train = False
        
        if coke_mode != "pqa":
            self.mask_head = False

        # todo: fix config loading (hardcode)
        if dataset.lower().startswith("w"):
            self.args.task = "wn18rr-ind_paths"
            self.args.vocab_path = "../../coke/CoKE/data/wn18rr-ind_paths/vocab.txt"
            self.args.vocab_size = 39986
            
            if max_len == -1:
                self.args.init_checkpoint = "../../coke/CoKE/output/output_wn18rr-ind_paths/models/step_16000"
                max_len = 7
            elif max_len == 5:
                self.args.init_checkpoint = "../../coke/CoKE/output/output_wn18rr-ind_paths_len3/models/step_6598"
            elif max_len in [4,6]:
                print(f"no model for the specified length {max_len}. Falling back to the universal model.")
                self.args.init_checkpoint = "../../coke/CoKE/output/output_wn18rr-ind_paths/models/step_6927"
                max_len = 7

        elif dataset.lower().startswith("f"):
            self.args.task = "fb15k237_paths"
            self.args.vocab_path = "../../coke/CoKE/data/fb15k237_paths/vocab.txt"
            self.args.vocab_size = 15020
            if max_len == -1:
                self.args.init_checkpoint = "../../coke/CoKE/output/output_fb15k237_paths/models/step_58462"
                max_len = 7
            elif max_len in [4,6]:
                self.args.init_checkpoint = f"../../coke/CoKE/output/output_fb15k237_paths_len{max_len-2}/models/step_106294"     
        self.args.use_cuda = True
        self.args.max_seq_len = max_len
        self.args.max_position_embeddings = max_len
        
        print(self.args)
        if not (self.args.do_train or self.args.do_predict):
            raise ValueError("For args `do_train` and `do_predict`, at "
                             "least one of them must be True.")
        if self.args.use_cuda:
            place = fluid.CUDAPlace(0)
            self.dev_count = fluid.core.get_cuda_device_count()
        else:
            place = fluid.CPUPlace()
            self.dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        self.exe = fluid.Executor(place)

        startup_prog = fluid.Program()

        # Init programs
        coke_config = init_coke_net_config(self.args, print_config=True)

        # Create model for prediction
        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, startup_prog):
            with fluid.unique_name.guard():
                self.embs, self.test_pyreader, _, self.fc_out, num_seqs = create_model(
                    pyreader_name='test_reader',
                    coke_config=coke_config,
                    args=self.args)

                if self.args.use_ema and 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(self.args.ema_decay)

                fluid.memory_optimize(self.test_prog, skip_opt_set=[self.fc_out.name, num_seqs.name])

        self.test_prog = self.test_prog.clone(for_test=True)

        self.exe.run(startup_prog)
        init_predict_checkpoint(self.args, self.exe, startup_prog)    

    ### GET EMBEDDING ###

    def get_embeddings(self):
        # fill data with idx of all elements of the vocab via one element sequences
        data = torch.tensor([[0, i, 0] for i in range(self.args.vocab_size)])  # add pad idx for both possible masked positions
        test_data_reader = self.get_data_reader(data, is_training=False, epoch=1, shuffle=False, dev_count=self.dev_count,
                                                vocab_size=self.args.vocab_size)
        self.test_pyreader.decorate_tensor_provider(test_data_reader.data_generator())

        embeddings_all = []

        self.test_pyreader.start()
        while True:
            try:
                # note: return a numpy array
                embeddings_all.append(self.exe.run(fetch_list=[self.embs.name], program=self.test_prog)[0])
            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break
        # collect all embedded *sequences*
        out = np.concatenate(embeddings_all, axis=0)

        # extract relevant position
        out = out[:, 1]
        print(out.shape)
        return out

    ### ACTUAL BATCH PREDICTION ###

    def get_predictions(self, batch_tensor):
        # Run prediction
        assert self.dev_count == 1, "During prediction, dev_count expects 1, current is %d" % self.dev_count
        test_data_reader = self.get_data_reader(batch_tensor, is_training=False,
                                                epoch=1, shuffle=False, dev_count=self.dev_count,
                                                vocab_size=self.args.vocab_size)
        self.test_pyreader.decorate_tensor_provider(test_data_reader.data_generator())

        total_fc_out = self.predict(test_data_reader.examples)

        logger.debug(">>Finish predicting at %s " % time.ctime())
        return total_fc_out

    def predict(self, all_examples):
        dataset = self.args.dataset
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)

        #sen_candli_dict, trivial_sen_set = _load_pathquery_eval_dict(self.args.sen_candli_file,
        #                                                           self.args.sen_trivial_file)
        # make predictions
        eval_i = 0
        step = 0
        # ideally only one batch per reader (one fc_out)
        total_fc_out = []

        self.test_pyreader.start()
        while True:
            try:
                # note: return a numpy array
                total_fc_out.append(self.exe.run(fetch_list=self.fc_out.name, program=self.test_prog)[0])
                step += 1
                if step % 10 == 0:
                    logger.info("Processing pathquery_predict step:%d example: %d" % (step, eval_i))
                _batch_len = total_fc_out[-1].shape[0]
                eval_i += _batch_len
            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break
        return np.concatenate(total_fc_out, axis=0)

        #logger.info("\n---------- Evaluation Performance --------------\n%s\n%s" %
        #            ("\t".join(["TASK", "MQ", "Hits@10"]), outs))

    def get_data_reader(self, batch_tensor, epoch, is_training, shuffle, dev_count, vocab_size):
        Reader = PathqueryTensorReader
        data_reader = Reader(
            vocab_path=self.args.vocab_path,
            data=batch_tensor,
            max_seq_len=self.args.max_seq_len,
            batch_size=self.args.batch_size,
            is_training=is_training,
            shuffle=shuffle,
            dev_count=dev_count,
            epoch=epoch,
            vocab_size=vocab_size,
            mask_head=self.mask_head)
        return data_reader

    def embed_path(self, sequences, use_labels=None, test_mode=None):
        '''
        Prepare given sequence and make prediction scores.
        :param sequences: full paths e, r, e ..., r, e
        :param use_labels: exists for compatibility reasons
        :param test_Mode: exists for compatibility reasons
        :return: prediction loss (0), predicted scores
        '''

        # todo: check if instead of flipping relations should be inverted
        e0 = sequences[:, :1]
        path_rel = sequences[:, 3::2]
        qr = sequences[:, 1:2]
        t = sequences[:, 2:3]
        e1 = sequences[:, -1:]
        #print(sequences.shape, e0.shape, rel.shape)
        #assert(path_rel.shape[1] == self.option.max_step_length)

        # flip to make tail prediction
        if self.mask_head:
            coke_seq = torch.cat([e0, rel, e1], dim=-1)
        else:
            rel_flip = torch.LongTensor([[self.rel2inv[r.item()] if r.item() not in [0,3]  else r.item() for r in r_seq] for r_seq in path_rel.flip(-1)])
            qr_flip = torch.LongTensor([[self.rel2inv[r.item()]] for r in qr])
            if self.coke_mode == "lp":
                coke_seq = torch.cat([rel_flip, t, qr_flip, e0], dim=-1)
            elif self.coke_mode == "anchor":
                coke_seq = torch.cat([e1, rel_flip, t, qr_flip, e0], dim=-1)
            elif self.coke_mode == "pqa":
                coke_seq = torch.cat([e1, rel_flip, qr_flip, e0], dim=-1)

        scores_np = self.get_predictions(coke_seq.detach().cpu().numpy())
        #print(scores_np.shape, self.item_embedding.num_embeddings)
        assert scores_np.shape[0] == sequences.shape[0]
        #assert scores_np.shape[1] == self.item_embedding.num_embeddings ## item embedding has extra "out of bert" tokens in kg_rl.py
        return 0, torch.from_numpy(scores_np)


def convert_ids_to_ids(target_vocab, source_vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        output.append(target_vocab[source_vocab[item]])
    return output

RawExample = collections.namedtuple("RawExample", ["token_ids", "mask_type"])

class PathqueryTensorReader(KBCDataReader):
    def __init__(self,
                 vocab_path,
                 data,
                 external_vocab_path=None,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1,
                 mask_head=True):
        self.mask_type = "MASK_HEAD" if mask_head else "MASK_TAIL"
        if external_vocab_path is not None:
            self.source_vocab = {v:k for k,v in load_vocab(external_vocab_path).items()}
        else:
            self.source_vocab = None
        KBCDataReader.__init__(self, vocab_path, data, max_seq_len,
                               batch_size, is_training, shuffle, dev_count,
                               epoch, vocab_size)


    def read_example(self, input_data):
        """Translate from other encoding to the model vocab"""
        examples = []
        for ids_source in input_data:
            assert len(ids_source) <= (self.max_seq_len + 1), \
                "Expecting at most [max_seq_len + 1]=%d tokens each line, current tokens %d" \
                % (self.max_seq_len + 1, len(ids_source))
            if self.source_vocab is not None:
                ids_target = convert_ids_to_ids(self.vocab, self.source_vocab, ids_source)
            else:
                ids_target = ids_source

            if len(ids_target) <= 0:
                continue
            examples.append(
                RawExample(
                    token_ids=ids_target, mask_type=self.mask_type)) # MASK_TYPES: "MASK_HEAD" or "MASK_TAIL"
            # if len(examples) <= 10:
            #     logger.info("*** Example ***")
            #     logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
            #     logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
        return examples

