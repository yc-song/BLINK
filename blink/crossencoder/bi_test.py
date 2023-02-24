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
import sys
sys.path.append('.')
from multiprocessing.pool import ThreadPool
from torch import optim
from tqdm import tqdm, trange
from collections import OrderedDict
import gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
import logging
import wandb
import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.crossencoder.mlp import MlpModel
# from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt

## Evaluation on test set
test_split = 2
data_path = "/home/jongsong/BLINK/models/zeshel/top64_candidates_base/"
fname = os.path.join(data_path, "test_raw_wo64.t7")
test_data = torch.load(fname)
context_input = test_data["context_vecs"]
candidate_input = test_data["candidate_vecs"]
label_input = test_data["labels"]

bi_val_accuracy = torch.mean((label_input == 0).float()).item()
bi_val_recall_4 = torch.mean((label_input <= 3).float()).item()
print(bi_val_accuracy, bi_val_recall_4)