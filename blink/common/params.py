# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using BLINK.
import argparse
import importlib
import os
import sys
import datetime
import json

with open('./blink/common/crossencoder_config.json') as f:
    config = json.load(f)



ENT_START_TAG = "[unused1]"
ENT_END_TAG = "[unused2]"
ENT_TITLE_TAG = "[unused3]"

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BlinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args:
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_blink_args=True, add_model_args=False, 
        description='BLINK parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            default = config["data_parallel"],
            type = str2bool,
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=config["top_k"], type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            default=config["zeshel"],
            type=str2bool,
            help="Whether the dataset is from zeroshot.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--layers",
            default=config["layers"],
            type=int,
            help="number of layers for mlp structure",
        )
        parser.add_argument(
            "--with_mlp", action="store_true", help="Whether to add mlp layers on top of Bi-encoder."
        )
        parser.add_argument(
            "--anncur", action="store_true", help="Whether to add mlp layers on top of Bi-encoder."
        )
        parser.add_argument(
            "--act_fn",
            default=config["act_fn"],
            help="softplus, sigmoid, tanh",
        )
        parser.add_argument(
            "--step_size",
            default=config["step_size"],
            type=int,
            help="number of layers for mlp structure",
        )
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=config["max_context_length"],
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=config["max_cand_length"],
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--path_to_mlpmodel",
            default=None,
            type=str,
            required=False,
            help="The full path to the mlp model to load. (in case of mlp-with-bert model)",
        )
        parser.add_argument(
            "--bert_model",
            default=config["bert_model"],
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--architecture",
            default=config["architecture"],
            type=str,
            help="mlp, bert, roberta, special_token, raw_context_text, mlp_with_bert",
        )
        parser.add_argument(
            "--loss_weight",
            default=config["loss_weight"],
            type=float,
            help="multi-task loss weight",
        )
        parser.add_argument(
            "--late_interaction",
            default=config["late_interaction"],
            type=str2bool,
            help="mlp, bert, roberta, special_token, raw_context_text, mlp_with_bert",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_false",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument("--context_key", default="context", type=str)
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            default= False,
            type= str2bool,
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default=config["data_path"],
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=config["output_path"],
            type=str,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )
        parser.add_argument(
            "--wandb",
            default=config["wandb"],
            type=str,
            help="wandb project name.",
        )
        parser.add_argument(
            "--resume",
            default=config["resume"],
            type=str2bool,
            help="wandb project name.",
        )

        parser.add_argument(
            "--run_id",
            default=config["run_id"],
            type=str,
            help="wandb project name.",
        )


        parser.add_argument(
            "--without_64",
            default=config["without_64"],
            type=str2bool,
            help="to include 64 or not",
        )
        parser.add_argument(
            "--timeout",
            default=config["timeout"],
            type=int,
            help="timeout minutes.",
        )

        parser.add_argument(
            "--decoder",
            default=config["decoder"],
            type=str2bool,
            help="decoder strucutre or not.",
        )
        parser.add_argument(
            "--split",
            default=config["split"],
            type=int,
            help="split dataset into N chunks. (because of out of memory)",
        )
        parser.add_argument(
            "--train_split",
            default=config["train_split"],
            type=int,
            help="N when train dataset splitted into N chunks. (because of out of memory)",
        )
        parser.add_argument(
            "--valid_split",
            default=config["valid_split"],
            type=int,
            help="N when valid dataset splitted into N chunks. (because of out of memory)",
        )
        parser.add_argument(
            "--test_split",
            default=config["test_split"],
            type=int,
            help="N when test dataset splitted into N chunks. (because of out of memory)",
        )

        


    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )

        parser.add_argument(
            "--scheduler_gamma",
            default=config["scheduler_gamma"],
            type=float,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--cos_similarity",
            action = "store_true",
            help="replace dot product with cos similarity in mlp-with-som module",
        )
        parser.add_argument(
            "--sampling",
            default=config["sampling"],
            type=str2bool,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--dot_product",
            default=config["dot_product"],
            type=str2bool,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--binary_loss",
            default=config["binary_loss"],
            type=str2bool,
            help="Binary cross entropy loss or multi class cross entropy loss.",
        )
        parser.add_argument(
            "--hard_negative",
            default=config["hard_negative"],
            type=str2bool,
            help="Random sampling or hard negative mining.",
        )
        parser.add_argument(
            "--num_samples",
            default=config["num_samples"],
            type=str2bool,
            help="number of samples.",
        )
        parser.add_argument(
            "--weight_decay",
            default=config["scheduler_gamma"],
            type=float,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--train_batch_size", default=config["train_batch_size"], type=int, 
            help="Total batch size for training."
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=config["learning_rate"],
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=config["num_train_epochs"],
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--dim_red",
            default=config["dim_red"],
            type=int,
            help="first dimension",
        )
        parser.add_argument(
            "--train_size",
            default=config["train_size"],
            type=int,
            help="dataset size of train set",
        )
        parser.add_argument(
            "--valid_size",
            default=config["valid_size"],
            type=int,
            help="dataset size of dev set",
        )
        parser.add_argument(
            "--test_size",
            default=config["test_size"],
            type=int,
            help="dataset size of test set",
        )
        parser.add_argument(
            "--patience",
            default=config["patience"],
            type=int,
            help="patience for early stopping.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=50, 
            help="Interval of loss printing",
        )
        parser.add_argument(
            "--positional_encoding", type=str2bool, default=config["positional_encoding"], 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=config["eval_interval"],
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=config["save_interval"], 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default=config["type_optimization"],
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", type=str2bool, default=False, 
            help="Whether to shuffle train data",
        )
        parser.add_argument(
            "--save", type=str2bool, default=config["save"], 
            help="directory to save models",
        )
        parser.add_argument(
            "--dim", type=int, default=1024
        )
        parser.add_argument(
            "--optimizer",
            default=config["optimizer"],
            type=str,
            help="optimizer.",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--eval_batch_size", default=config["eval_batch_size"], type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for cached candidate pool (id tokenization of candidates)",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for cached candidate encoding",
        )
        parser.add_argument(
            "--cand_cls_path",
            default=None,
            type=str,
            help="Path for cached candidate cls encoding",
        )

