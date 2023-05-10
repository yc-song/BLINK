from torchsummary import summary
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np
import pandas as pd
import csv
import sys
sys.path.append('.')
from torchprofile import profile_macs
from multiprocessing.pool import ThreadPool
from torch import optim
from tqdm import tqdm, trange
from collections import OrderedDict
import gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BertTokenizer
import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import SOMRanker, CrossEncoderRanker, load_crossencoder, MlpwithBiEncoderRanker, MlpwithSOMRanker
import logging
import wandb
import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.crossencoder.mlp import MlpModel
from blink.biencoder.biencoder import BiEncoderRanker
from ptflops import get_model_complexity_info
from torchinfo import summary

# from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def main(params):
    gc.collect()
    nested_break=False
    # wandb.init(project=params["wandb"], config=parser, resume="must", id=<original_sweeps_run_id>)
    epoch_idx_global = 0
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
    previous_step = 0
        # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    train_split = params["train_split"]

    # Init model
    if params["architecture"]=="mlp":
        reranker= MlpModel(params)
    elif params["architecture"] == "mlp_with_bert":
        reranker = MlpwithBiEncoderRanker(params)
    elif params["architecture"] == "mlp_with_som":
        reranker = MlpwithSOMRanker(params)
    elif params["architecture"] == "som":
        reranker = SOMRanker(params)
    elif params["architecture"] == "bi_encoder":
        reranker = BiEncoderRanker(params)
    else:
        reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    model.eval()
    reranker = reranker.to(device)
    # baseline
    # input = torch.randint(1, 3, (64, 256)).to(device)
    # mlp-with-bert
    # input = torch.randint(1, 3, (64, 128))
    # mlp
    input = torch.rand(64, 2, 768).to(device)
    # mlp-with-som
    # input = torch.rand(1, 64, 2, 128, 768).to(device)

    summary(model,  input_data = input)


    
if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)