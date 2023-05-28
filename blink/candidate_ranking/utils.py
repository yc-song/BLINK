# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import io
import sys
import json
import torch
import logging
from scipy.stats import rankdata
import numpy as np

from collections import OrderedDict
# CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
from tqdm import tqdm

from blink.candidate_ranking.bert_reranking import BertReranker
from blink.biencoder.biencoder import BiEncoderRanker


def read_dataset(dataset_name, preprocessed_json_data_parent_folder, debug=False):
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)

    samples = []
    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            samples.append(json.loads(line.strip()))
            if debug and len(samples) > 200:
                break
            # samples.append(json.loads(line.strip()))
            # if debug and len(samples) > 200:
            #     break
    return samples


def filter_samples(samples, top_k, gold_key="gold_pos"):
    if top_k == None:
        return samples

    filtered_samples = [
        sample
        for sample in samples
        if sample[gold_key] > 0 and sample[gold_key] <= top_k
    ]
    return filtered_samples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop() 


def eval_precision_bm45_dataloader(dataloader, ks=[1, 5, 10], number_of_samples=None):
    label_ids = torch.cat([label_ids for _, _, _, label_ids, _ in dataloader])
    label_ids = label_ids + 1
    p = {}

    for k in ks:
        p[k] = 0

    for label in label_ids:
        if label > 0:
            for k in ks:
                if label <= k:
                    p[k] += 1
    for k in ks:
        if number_of_samples is None:
            p[k] /= len(label_ids)
        else:
            p[k] /= number_of_samples

    return p


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    # print("outputs", outputs)
    # twomax = np.partition(out, -2)[:, -2:].T
    # default = -1
    # outputs=np.where(twomax[0] != twomax[1], np.argmax(out, -1), default)
    # print("labels", labels)
    # print("max outputs", outputs)
    # print("label", labels)
    # print("ranking", np.take(idx_array, labels))
    return np.sum(outputs == labels), outputs == labels

def accuracy_label64(out, labels, label_64 = True):
    if label_64 == True:
        mask= (labels[:]==64)
    else:
        mask= (labels[:]!=64)

    outputs = np.argmax(out, axis=1)
    # print("outputs", outputs)
    # print("labels", labels)
    # print(np.sum(outputs[mask]==labels[mask]), np.sum(mask==True))
    # return (total success of prediction, total # of label 64)
    return np.sum(outputs[mask]==labels[mask]), np.sum(mask==True)

def mrr_label64(out, labels):
    mask= (labels[:]==64)
    masked_labels=labels[mask]
    idx_array = rankdata(-out[mask], axis=1, method='min')
    rank = np.take_along_axis(idx_array, masked_labels[:,None], axis=1)
    return np.sum(1/rank)


def mrr(out, labels, train = True, input = None): #implement mean reciprocal rank
    idx_array = rankdata(-out, axis=1, method='min')
    
    rank = np.take_along_axis(idx_array, labels[:, None], axis=1)
    #print out best and worst cases
    # if train == False: 
    #     ranks=[1,65]
    #     for r in ranks:
    #         mask= (rank[:,0]==r)
    #         mask_outputs= outputs[mask]
    #         mask_labels=labels[mask]
    #         if np.any(mask):
    #             print("\n rank", r)
    #             print("logits", out[mask])
    #             print("input shape", input[mask,mask_outputs,:,:].shape)
    #             print("input for prediction", input[mask,mask_outputs,:,:][0])
    #             print("input for gold", input[mask,mask_labels,:,:][0])


    return np.sum(1/rank)

def recall(out, labels, k=4):
    idx_array = rankdata(-out, axis=1, method='min')
    rank = np.take_along_axis(idx_array, labels[:,None], axis=1)
    # print("labels_recall", labels[:,None])
    # print("idx_array", idx_array)
    # print("rank", rank.T)
    # print("rank", rank)
    # print(rank<=k)
    # print(np.count_nonzero(rank<=k))
    return np.count_nonzero(rank<=k)
    
def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def save_model(model, tokenizer, output_dir, epoch = None, optimizer = None):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    if epoch is not None:
        torch.save({"state_dict":model_to_save.state_dict(),\
        "epoch": epoch,\
        "optimizer_state_dict": optimizer.state_dict(),\
        }, output_model_file)
    else:
        torch.save(model.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def load_model(most_recent_file, reranker, optimizer):
    checkpoint = torch.load(most_recent_file)
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        if not k.startswith('module'):
            name = "model.module."+k
            new_state_dict[name] = v
        elif not k.startswith('model'):
            name = "model."+k
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    reranker.load_state_dict(new_state_dict)
    epoch_idx_global = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return reranker, epoch_idx_global, optimizer

def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('Blink')
    logger.setLevel(10)
    return logger


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def get_reranker(parameters):
    return BertReranker(parameters)


def get_biencoder(parameters):
    return BiEncoderRanker(parameters)
