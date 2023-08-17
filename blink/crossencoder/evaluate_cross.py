# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BertTokenizer
import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import SOMRanker, ExtendSingleRanker, ExtendExtensionRanker, ExtendMultiRanker, AdditiveScoreRanker, MultiplicativeScoreRanker, CrossEncoderRanker, load_crossencoder, MlpwithBiEncoderRanker, MlpwithSOMRanker
from blink.biencoder.biencoder import BiEncoderRanker
import logging
import wandb
import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer_copy import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.crossencoder.mlp import MlpModel
# from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt
logger = None
scheduler_architecture = [
    "mlp_with_bert",
    "baseline",
    "raw_context_text",
    "special_token",
    "som",
    "mlp_with_som_finetuning",
    "extend_extension",
    "extend_multi",
    "extend_single"
]
# Below are architectures which causes OOM issue when function modify() is not called appropraitely
oom_architecture= [
    "mlp_with_som",
    "som",
    "mlp",
    "multiplicative_score",
    "additive_score",
    "extend_extension",
    "extend_multi",
    "extend_single"

] 
# mlp_based_architecture and BERT_bsaed_architecture use different optimizer call
mlp_based_architecture = [
    "mlp",
    "mlp_with_som",
    "multiplicative_score",
    "additive_score",
    "extend_extension",
    "extend_multi",
    "extend_single"

]
bert_based_architecture = [
    "mlp_with_bert",
    "raw_context_text",
    "special_token",
    "baseline",
    "mlp_with_som_finetuning"
]

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))
def distribution_sample(probs, num_cands, device):
    num_nonzero = (probs != 0).sum(1)
    less_indices = (num_nonzero < num_cands)
    min_num_nonzero = num_nonzero.min().item()
    if min_num_nonzero < num_cands:
        candidates = torch.zeros((probs.size(0), num_cands)).long().to(device)
        candidates[less_indices] = torch.multinomial(probs[less_indices],
                                                     num_cands,
                                                     replacement=True)
        if candidates[less_indices].size(0) != probs.size(0):
            candidates[~less_indices] = torch.multinomial(probs[~less_indices],
                                                          num_cands,
                                                          replacement=False)
    else:
        candidates = torch.multinomial(probs, num_cands, replacement=False)
    return candidates
def modify(context_input, candidate_input, params, world, idxs, mode = "train", wo64 = True, label_input = None, scores = None, top_k = 64, self_evaluate_again = False):
    device = torch.device('cuda')
    # get values of candidates first
    # candidate_input = candidate_input.to(device)
    assert mode == "train" or mode == "valid" or mode == "test"
    # print(mode, idxs, idxs.shape)
    if mode == "train":
        if params["hard_negative"] or params["self_evaluate"]:
            labels = idxs[torch.arange(idxs.size(0)),label_input] # (128)
            probs = scores.softmax(dim = -1)
            ### Original Code ####
            hard_cands = idxs.gather(1, distribution_sample(probs, top_k, device)) # (128, 64) 
            result = [None]*labels.shape[0]
            for i in range(labels.shape[0]):
                result[i] = [labels[i].item()] + [cand.item() for cand in hard_cands[i] if cand != labels[i]]
                result[i] = result[i][:top_k]
            idxs = torch.tensor(result)
        else: 
            mask_eval_cands = label_input < params["train_cands"]
            context_input= context_input[mask_eval_cands]
            world = world[mask_eval_cands]
            idxs = idxs[mask_eval_cands, :params["train_cands"]] 
            label_input = label_input[mask_eval_cands] 
            #######################
            #### For unit testing ####
            # print(label_input)
            # hard_cands = idxs.cpu().gather(1, torch.arange(64).unsqueeze(0).expand(labels.size(0), -1)) # (128, 64) 
            # result = [None]*labels.shape[0]
            # for i in range(labels.shape[0]):
            #     if label_input[i] < params["top_k"]:
                    
                
        #     result[i] = [labels[i].item()] + [cand.item() for cand in hard_cands[i] if cand != labels[i]]
        #     result[i] = result[i][:top_k]             
        # print("after", mode, idxs, idxs.shape)
    candidate_input = candidate_input[idxs.cpu()].to(device)
    top_k=params["top_k"]
    ## context_input shape: (Size ,1024) e.g.  (10000, 1024)
    ## candidate_input shape: (Size, 65, 1024) e.g. (10000, 65, 1024)
    if wo64 == False:
        if mode == "train" or mode == "valid":
            context_input=context_input.unsqueeze(dim=1).expand(-1,top_k+1,-1)
            if params["architecture"]=="mlp" or params["architecture"]=="special_token":
                new_input=torch.stack((context_input,candidate_input),dim=2) # shape: (Size, 65, 2, 1024) e.g. (10000, 65, 2 , 1024)
            elif params["architecture"]== "raw_context_text":
                new_input=torch.cat((context_input,candidate_input),dim=2)
            # print(new_input.shape)
            return new_input
        elif mode == "test":
            context_input=context_input.unsqueeze(dim=1).expand(-1,top_k,-1)
            if params["architecture"]=="mlp" or params["architecture"]=="special_token":
                new_input=torch.stack((context_input,candidate_input),dim=2) # shape: (Size, 65, 2, 1024) e.g. (10000, 65, 2 , 1024)
            elif params["architecture"]== "raw_context_text":
                new_input=torch.cat((context_input,candidate_input),dim=2)
            # print(new_input.shape)
            # print(new_input.shape)
            return new_input
    else: 
        if params["architecture"] == "mlp_with_som" or params["architecture"] == "som":
            context_input =  context_input.unsqueeze(dim=1).expand(-1,top_k,-1,-1).to(device)
            new_input = torch.stack((context_input, candidate_input), dim = 2   )
        elif params["architecture"] == "extend_single":
            context_input =  context_input.unsqueeze(dim=1).expand(-1,top_k,-1,-1).to(device)
            new_input = torch.cat((context_input, candidate_input.unsqueeze(-2)), dim = 2  )
        elif params["architecture"] == "extend_extension":
            new_input = torch.cat([context_input.unsqueeze(-2), candidate_input], dim = -2) #(Batch size, top_k+1, BERT embedding dimension)
        elif params["architecture"] == "extend_multi":
            new_input = torch.cat([context_input, candidate_input], dim = -2) #(Batch size, top_k+1, BERT embedding dimension)

        else:
            context_input=context_input.unsqueeze(dim=1).expand(-1,top_k,-1).to(device)
            if params["architecture"] =="mlp" or params["architecture"] =="additive_score" or params["architecture"] =="multiplicative_score" or params["architecture"]=="special_token" or params["architecture"]=="mlp_with_bert"  or params["architecture"]=="mlp_with_som_finetuning":
                new_input=torch.stack((context_input,candidate_input),dim=2) # shape: (Size, 65, 2, 1024) e.g. (10000, 65, 2 , 1024)
            elif params["architecture"] == "raw_context_text" or params["architecture"]=="baseline":
                new_input=torch.cat((context_input,candidate_input),dim=2)
        return new_input, label_input
        


def evaluate(reranker, eval_dataloader, device, logger, context_length, candidate_input, zeshel=False, silent=True, mode = "valid", input = None, wo64 = True, hard_entire = False, recall_k = 4):
    reranker.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0


    eval_mrr = 0.0
    if wo64 == False:
        eval_accuracy_64 = 0.0
        eval_accuracy_not64 = 0.0
        total_labels_64 = 0.0
        total_labels_not64 =0.0
        eval_mrr_64 = 0.0
        total_mrr_64 =0.0

    nb_eval_examples = 0
    nb_eval_steps = 0
    eval_recall=0
    
    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    # all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        if params["top_k"]>100 or params["architecture"] in oom_architecture:
            world = batch[2]
            idxs = batch[3]
            if mode == "valid" and not hard_entire:
                mask_eval_cands = label_input < params["eval_cands"]
                context_input= context_input[mask_eval_cands]
                world = world[mask_eval_cands]
                idxs = idxs[mask_eval_cands, :params["eval_cands"]] 
                label_input = label_input[mask_eval_cands] 
            context_input, label_input = modify(context_input, candidate_input, params, world, idxs, mode = "valid", wo64 = params["without_64"], label_input = label_input)

        with torch.no_grad():
            if hard_entire:
                logits = reranker(context_input, label_input, context_length, evaluate = True, hard_entire = True)
            elif params["dot_product"]:
                eval_loss, logits, _, _ = reranker(context_input, label_input, context_length, evaluate = True)
            else:
                eval_loss, logits = reranker(context_input, label_input, context_length, evaluate = True)
        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        
        eval_recall += utils.recall(logits, label_ids, k = recall_k)
        # print("recall", eval_recall)
        eval_mrr+=utils.mrr(logits, label_ids, input = context_input)
        eval_accuracy += tmp_eval_accuracy
        if wo64 == False:
            tmp_eval_accuracy_64, tmp_labels_64 = utils.accuracy_label64(logits, label_ids)
            tmp_eval_accuracy_not64, tmp_labels_not64 = utils.accuracy_label64(logits, label_ids, label_64=False)
            eval_mrr_64+=utils.mrr_label64(logits, label_ids)
            eval_accuracy_64 += tmp_eval_accuracy_64
            eval_accuracy_not64 += tmp_eval_accuracy_not64
            total_labels_64 += tmp_labels_64
            total_labels_not64 += tmp_labels_not64

        # print("accuracy", eval_accuracy)
        if step == 0: all_logits = torch.tensor(logits)
        else: all_logits = torch.cat((all_logits, torch.tensor(logits)), dim = 0)
        # all_logits.append(logits.tolist())
        

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1
    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        if wo64 == False:
            normalized_eval_accuracy_64 = eval_accuracy_64 / total_labels_64
            normalized_eval_accuracy_not64 = eval_accuracy_not64 / total_labels_not64
            eval_mrr_64 = eval_mrr_64 / total_labels_64
        eval_recall/=nb_eval_examples
        eval_mrr /= nb_eval_examples

    if zeshel:
        macro = 0.0
        num = 0.0 
        logger.info("By world: {}, {}".format(acc, tot))

        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    logger.info("MRR: %.5f" % eval_mrr)
    if params["without_64"] == False:
        logger.info("MRR @ label 64: %.5f" % eval_mrr_64)
        logger.info("Micro accuracy @ label 64: %.5f" % normalized_eval_accuracy_64)

    logger.info("Recall@{}: %.5f".format(recall_k) % eval_recall)
    if num != 0:
        results["macro_accuracy"] = macro/num
    else:
        results["macro_accuracy"] = 0
    results["normalized_accuracy"] = normalized_eval_accuracy
    
    results["logits"] = all_logits
    results["mrr"]=eval_mrr
    if params["without_64"] == False:
        results["mrr_64"]=eval_mrr_64
        results["normalized_accuracy_64"]=normalized_eval_accuracy_64
        results["normalized_accuracy_not64"]=normalized_eval_accuracy_not64
    results["recall"]=eval_recall
    return results

def recall_evaluate(reranker, eval_dataloader, device, logger, context_length, candidate_input, zeshel=False, silent=True, mode = "valid", input = None, wo64 = True, hard_entire = False, recall_k = 4):
    reranker.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    eval_recall = 0.0
    eval_mrr = 0.0

    nb_eval_examples = 0
    nb_eval_steps = 0
    
    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    # all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        if params["top_k"]>100 or params["architecture"] in oom_architecture:
            world = batch[2]
            idxs = batch[3] 
            context_input, label_input = modify(context_input, candidate_input, params, world, idxs, mode = "valid", wo64 = params["without_64"], label_input = label_input)

        with torch.no_grad():
            if params["self_evaluate"] and hard_entire:
                logits = reranker(context_input, label_input, context_length, evaluate = True, hard_entire = True)
            elif params["dot_product"]:
                eval_loss, logits, _, _ = reranker(context_input, label_input, context_length, evaluate = True)
            else:
                eval_loss, logits = reranker(context_input, label_input, context_length, evaluate = True)
        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        
        eval_recall += utils.recall(logits, label_ids, 64)
        # print("recall", eval_recall)
        eval_mrr+=utils.mrr(logits, label_ids, input = context_input)
        eval_accuracy += tmp_eval_accuracy
        if wo64 == False:
            tmp_eval_accuracy_64, tmp_labels_64 = utils.accuracy_label64(logits, label_ids)
            tmp_eval_accuracy_not64, tmp_labels_not64 = utils.accuracy_label64(logits, label_ids, label_64=False)
            eval_mrr_64+=utils.mrr_label64(logits, label_ids)
            eval_accuracy_64 += tmp_eval_accuracy_64
            eval_accuracy_not64 += tmp_eval_accuracy_not64
            total_labels_64 += tmp_labels_64
            total_labels_not64 += tmp_labels_not64

        # print("accuracy", eval_accuracy)
        if step == 0: all_logits = torch.tensor(logits)
        else: all_logits = torch.cat((all_logits, torch.tensor(logits)), dim = 0)
        # all_logits.append(logits.tolist())
        

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1
    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        if wo64 == False:
            normalized_eval_accuracy_64 = eval_accuracy_64 / total_labels_64
            normalized_eval_accuracy_not64 = eval_accuracy_not64 / total_labels_not64
            eval_mrr_64 = eval_mrr_64 / total_labels_64
        eval_recall/=nb_eval_examples
        eval_mrr /= nb_eval_examples

    if zeshel:
        macro = 0.0
        num = 0.0 
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    logger.info("MRR: %.5f" % eval_mrr)
    if params["without_64"] == False:
        logger.info("MRR @ label 64: %.5f" % eval_mrr_64)
        logger.info("Micro accuracy @ label 64: %.5f" % normalized_eval_accuracy_64)

    logger.info("Recall@4: %.5f" % eval_recall)
    if num != 0:
        results["macro_accuracy"] = macro/num
    else:
        results["macro_accuracy"] = 0
    results["normalized_accuracy"] = normalized_eval_accuracy
    
    results["logits"] = all_logits
    results["mrr"]=eval_mrr
    if params["without_64"] == False:
        results["mrr_64"]=eval_mrr_64
        results["normalized_accuracy_64"]=normalized_eval_accuracy_64
        results["normalized_accuracy_not64"]=normalized_eval_accuracy_not64
    results["recall"]=eval_recall
    return results


def get_optimizer(model, params, architecture = None):
    if architecture is None:
        return get_bert_optimizer(
            [model],
            params["type_optimization"],
            params["bert_lr"],
            fp16=params.get("fp16"),
        )
    elif architecture == "mlp":
        mlp_layers = ['fc.weight', 'layers']
        parameters_mlp = []
        parameters_mlp_names = []
        for n, p in model.named_parameters():
            if any(t in n for t in mlp_layers):
                parameters_mlp.append(p)
                parameters_mlp_names.append(n)
        print('The following parameters will be optimized WITHOUT decay:')
        print(parameters_mlp_names[:5])
        optimizer_grouped_parameters = [
        {'params': parameters_mlp, 'weight_decay': 0.01}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=params["weight_decay"], lr=params["learning_rate"])
        return optimizer



def get_scheduler(params, optimizer, len_train_data, logger, last_epoch = -1):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_train_steps
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    gc.collect()
    nested_break=False
    # wandb.init(project=params["wandb"], config=parser, resume="must", id=<original_sweeps_run_id>)
    epoch_idx_global = 0
    previous_step = 0
    optimizer_mlp = None
        # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    train_split = params["train_split"]
    for i in range(train_split):
        if train_split == 1:
            fname = os.path.join(params["data_path"], "train_{}.t7".format(params["architecture"]))
        else:
            fname = os.path.join(params["data_path"], "train_{}_{}.t7".format(params["architecture"], i))
        if not os.path.isfile(fname):
            if params["architecture"] == "mlp":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("special_tokens"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("special_tokens", i))
            elif params["architecture"] == "special_tokens":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "som":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp_with_som"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp_with_som", i))
            elif params["architecture"] == "mlp_with_som_finetuning":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp_with_bert"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp_with_bert", i))
            elif params["architecture"] == "baseline":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp_with_bert"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp_with_bert", i))
            elif params["architecture"] == "multiplicative_score":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "additive_score":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_extension":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_multi":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_single":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("extend_multi"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("extend_multi", i))


        if i == 0:
            train_data = torch.load(fname,  map_location=torch.device('cpu'))
            # train_data = torch.load(fname)
            context_input_train = train_data["context_vecs"][:params["train_size"]]
            candidate_input_train = train_data["candidate_vecs"]
            idxs_train = train_data["indexes"][:params["train_size"]]
            label_input_train = train_data["labels"][:params["train_size"]]
            bi_encoder_score_train = train_data["nn_scores"][:params["train_size"]]
            if params["zeshel"]:
                src_input_train = train_data['worlds'][:params["train_size"]]
        else:
            # fname2 = os.path.join(params["data_path"], "train2.t7")
            train_data = torch.load(fname)
            context_input_train = torch.cat((context_input_train, train_data["context_vecs"][:params["train_size"]]), dim = 0)
            idxs_train = torch.cat((idxs_train, train_data["indexes"][:params["train_size"]]), dim = 0)
            label_input_train = torch.cat((label_input_train, train_data["labels"][:params["train_size"]]), dim = 0)
            bi_encoder_score_train = torch.cat((bi_encoder_score_train, train_data["nn_scores"][:params["train_size"]]), dim = 0)
            if params["zeshel"]:
                src_input_train = torch.cat((src_input_train, train_data['worlds'][:params["train_size"]]), dim = 0)
    if not params["hard_negative"] and not params["self_evaluate"]:
        mask_eval_cands = label_input_train < params["train_cands"]
        context_input_train= context_input_train[mask_eval_cands]
        src_input_train = src_input_train[mask_eval_cands]
        idxs_train = idxs_train[mask_eval_cands, :params["train_cands"]] 
        label_input_train = label_input_train[mask_eval_cands] 
        bi_encoder_score_train =  bi_encoder_score_train[mask_eval_cands]
    # Init model
    if params["architecture"]=="mlp":
        reranker= MlpModel(params)
    elif params["architecture"] == "mlp_with_bert":
        reranker = MlpwithBiEncoderRanker(params)
    elif params["architecture"] == "mlp_with_som":
        reranker = MlpwithSOMRanker(params)
    elif params["architecture"] == "som":
        reranker = SOMRanker(params)
    elif params["architecture"] == "baseline":
        reranker = CrossEncoderRanker(params)
    elif params["architecture"] == "mlp_with_som_finetuning":
        reranker = MlpwithBiEncoderRanker(params)
    elif params["architecture"] == "multiplicative_score":
        reranker = MultiplicativeScoreRanker(params)
    elif params["architecture"] == "additive_score":
        reranker = AdditiveScoreRanker(params)
    elif params["architecture"] == "extend_extension":
        reranker = ExtendExtensionRanker(params)
    elif params["architecture"] == "extend_multi":
        reranker = ExtendMultiRanker(params)
    elif params["architecture"] == "extend_single":
        reranker = ExtendSingleRanker(params)
    else:
        reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    if params["architecture"] in mlp_based_architecture:
        if params["optimizer"]=="Adam":
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        elif params["optimizer"]=="AdamW":
            optimizer = optim.AdamW(model.parameters(), weight_decay=params["weight_decay"], lr=params["learning_rate"], betas = (0.9, 0.98), eps = 1e-9)

        elif params["optimizer"]=="SGD":
            optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"], weight_decay=params["weight_decay"])
        elif params["optimizer"]=="RMSprop":
            optimizer = optim.RMSprop(model.parameters(),lr=params["learning_rate"])
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["scheduler_gamma"])
        # scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.9, min_lr=0.000001)
    elif params["architecture"] in bert_based_architecture:
        optimizer = get_optimizer(model, params)
        if params["architecture"] == "mlp_with_bert":
            optimizer_mlp = get_optimizer(model, params, architecture = "mlp")
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["scheduler_gamma"])


    if params["architecture"]=="mlp":
        print("model architecture", model.layers)
    # if you want to resume the training, set "resume" true and specify "run_id"
    if params["folder_path"] is not None:
        most_recent_file = params["folder_path"]
    if params["resume"]==True:
        if params["folder_path"] is not None:
            most_recent_file = params["folder_path"]
        else: 
            print('*****************')
            folder_path="models/zeshel/crossencoder/{}/{}/".format(params["architecture"], params["run_id"])
            each_file_path_and_gen_time = []
            ## Getting newest file
            for each_file_name in os.listdir(folder_path):
                each_file_path = folder_path + each_file_name
                if os.path.isfile(each_file_path):
                    each_file_gen_time = os.path.getctime(each_file_path)
                    each_file_path_and_gen_time.append(
                        (each_file_path, each_file_gen_time)
                    )
            most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
        run = wandb.init(project=params["wandb"], config=params, resume="must", id=params["run_id"])
        print("file loaded:", most_recent_file)
        checkpoint = torch.load(most_recent_file)
        reranker.load_state_dict(checkpoint['model_state_dict'])
        epoch_idx_global = checkpoint['epoch']
        previous_step = checkpoint['step']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        run = wandb.init(project=params["wandb"], config=params)
    config = wandb.config
    model_output_path = params["output_path"]+params["architecture"]+"/"+run.id
    if not params["resume"]:
        if not os.path.exists(model_output_path+"training_params"):
            os.makedirs(model_output_path+"/training_params")
        with open(os.path.join(model_output_path, "training_params/training_params.json"), 'w') as outfile:
            json.dump(params, outfile)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    if not os.path.exists(params["output_path"]+params["architecture"]+"/"+run.id+"/Epochs"): 
        os.makedirs(params["output_path"]+params["architecture"]+"/"+run.id+"/Epochs")

    logger = utils.get_logger(params["output_path"])
    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu
    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]


    # print(label_input)
    bi_train_mrr=torch.mean(1/(label_input_train+1)).item()
    bi_train_accuracy = torch.mean((label_input_train == 0).float()).item()
    bi_train_recall_4 = torch.mean((label_input_train <= 3).float()).item()


    logger.info("train bi-encoder mrr: {}".format(bi_train_mrr))
    wandb.log({
    "mrr/train_bi-encoder_mrr":bi_train_mrr,
    "acc/train_bi-encoder_accuracy": bi_train_accuracy,
    "recall/train_bi-encoder_recall@4": bi_train_recall_4,


    })

    if params["debug"]:
        max_n = 200
        context_input_train = context_input_train[:max_n]
        candidate_input_train = candidate_input_train[:max_n]
        src_input_train = src_input_train[:max_n]
        label_input_train = label_input_train[:max_n]
        bi_encoder_score_train = bi_encoder_score_train[:max_n]
        idxs_train = idxs_train[:max_n]

    if params["top_k"]<100 and not params["architecture"] in oom_architecture:
        context_input_train, _ = modify(context_input_train, candidate_input_train, params, src_input_train, idxs_train, mode = "train", wo64 = params["without_64"])
    if params["zeshel"]:
        train_tensor_data = TensorDataset(context_input_train, label_input_train, src_input_train, idxs_train, bi_encoder_score_train,)
        
    else:
        train_tensor_data = TensorDataset(context_input_train, label_input_train, src_input_train, idxs_train, bi_encoder_score_train)


    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )

    if params["architecture"] in scheduler_architecture:
        scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    valid_split = params["valid_split"]
    for i in range(valid_split):
        if valid_split == 1:
            fname = os.path.join(params["data_path"], "valid_{}.t7".format(params["architecture"]))
        else:
            fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format(params["architecture"], i))
        if not os.path.isfile(fname):
            if params["architecture"] == "mlp":
                if valid_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("special_tokens"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("special_tokens", i))
            elif params["architecture"] == "special_tokens":
                if valid_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "som":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp_with_som"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp_with_som", i))
            elif params["architecture"] == "baseline":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp_with_bert"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp_with_bert", i))
            elif params["architecture"] == "mlp_with_som_finetuning":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp_with_bert"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp_with_bert", i))
            elif params["architecture"] == "multiplicative_score":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "additive_score":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_extension":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_single":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "valid_{}.t7".format("extend_multi"))
                else:
                    fname = os.path.join(params["data_path"], "valid_{}_{}.t7".format("extend_multi", i))

        if i == 0:
            valid_data = torch.load(fname)
            context_input_valid = valid_data["context_vecs"][:params["valid_size"]]
            candidate_input_valid = valid_data["candidate_vecs"]
            idxs_valid = valid_data["indexes"][:params["valid_size"]]
            label_input_valid = valid_data["labels"][:params["valid_size"]]
            bi_encoder_score_valid = valid_data["nn_scores"][:params["valid_size"]]
            if params["zeshel"]:
                src_input_valid = valid_data['worlds'][:params["valid_size"]]
        else:
            # fname2 = os.path.join(params["data_path"], "train2.t7")
            valid_data = torch.load(fname)
            context_input_valid = torch.cat((context_input_valid, valid_data["context_vecs"][:params["valid_size"]]), dim = 0)
            idxs_valid = torch.cat((idxs_valid, valid_data["indexes"][:params["valid_size"]]), dim = 0)
            label_input_valid = torch.cat((label_input_valid, valid_data["labels"][:params["valid_size"]]), dim = 0)
            bi_encoder_score_valid = torch.cat((bi_encoder_score_valid, valid_data["nn_scores"][:params["valid_size"]]), dim = 0)
            if params["zeshel"]:
                src_input_valid = valid_data['worlds'][:params["valid_size"]]


    bi_val_mrr=torch.mean(1/(label_input_valid+1)).item()
    bi_val_accuracy = torch.mean((label_input_valid == 0).float()).item()
    bi_val_recall_4 = torch.mean((label_input_valid <= 3).float()).item()
    wandb.log({
    "mrr/val_bi-encoder_mrr":bi_val_mrr,
    "acc/val_bi-encoder_accuracy": bi_val_accuracy,
    "recall/val_bi-encoder_recall@4": bi_val_recall_4,
    })


    if params["debug"]:
        max_n = 200
        context_input_valid = context_input_valid[:max_n]
        candidate_input_valid = candidate_input_valid[:max_n]
        label_input_valid = label_input_valid[:max_n]
        bi_encoder_score_valid = bi_encoder_score_valid[:max_n]
        idxs_valid = idxs_valid[:max_n]
        src_input_valid = src_input_valid[:max_n]
    # print("valid context", context_input.shape)
    # print("valid candidate", candidate_input.shape)
    if params["top_k"]<100 and not params["architecture"] in oom_architecture:
        context_input_valid, _ = modify(context_input_valid, candidate_input_valid, params, src_input_valid, idxs_valid, mode = "valid", wo64=params["without_64"])
    # print("valid modify", context_input.shape)
    valid_tensor_data = TensorDataset(context_input_valid, label_input_valid, src_input_valid, idxs_valid)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    #evaluate before training
    if not params["resume"]:
        logger.info("Evaluation on the development dataset")
        val_acc=evaluate(
            reranker,
            valid_dataloader,
            candidate_input = candidate_input_valid,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
            wo64=params["without_64"]
        )
        wandb.log({
            "acc/val_acc": val_acc["normalized_accuracy"],
            "acc/val_acc(macro)": val_acc["macro_accuracy"]
            })
        val_acc_recall=evaluate(
            reranker,
            valid_dataloader,
            candidate_input = candidate_input_valid,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
            mode = "valid",
            wo64=params["without_64"],
            recall_k = 64, 
            hard_entire = True
        )
        wandb.log({
            "entire_instances/val_acc":val_acc_recall["normalized_accuracy"],
            "entire_instances/val_acc(macro)":val_acc_recall["macro_accuracy"],
            "entire_instances/val_recall":val_acc_recall["recall"],
        })

    number_of_samples_per_dataset = {}

    time_start = time.time()
    # utils.write_to_file(
    #     os.path.join(model_output_path, "training_params.json"), str(params)
    # )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )
    # optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"])
    # if params["optimizer"]=="Adam":
    #     optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        # print(optimizer)

    # elif params["optimizer"]=="SGD":
    #     optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"])
    # elif params["optimizer"]=="RMSprop":
        # optimizer = optim.RMSprop(model.parameters(), lr=params["learning_rate"])
 
    if params["architecture"] in scheduler_architecture:
        # when training resumes, use scheduler.step() multiple times to get aligned with previous run
        optimizer.step()
        if optimizer_mlp is not None:
            optimizer_mlp.step()
        for i in range(int(len(train_tensor_data) / params["train_batch_size"] / params["gradient_accumulation_steps"]) * epoch_idx_global + int(previous_step / params["gradient_accumulation_steps"])):
            scheduler.step()
    best_epoch_idx = -1
    best_score = -1
    best_score_macro = -1
    #Early stopping variables
    patience=params["patience"]
    last_acc=1
    trigger_times=-1
    print_interval=params["print_interval"]
    num_train_epochs = params["num_train_epochs"]
    print("epoch_idx_global", epoch_idx_global)
    for epoch_idx in trange(epoch_idx_global, int(num_train_epochs), desc="Epoch"):
        if params["self_evaluate"]:
            logger.info("Getting score distribution")
            train_logits=evaluate(
                reranker,
                train_dataloader,
                candidate_input = candidate_input_train,
                device=device,
                logger=logger,
                context_length=context_length,
                zeshel=params["zeshel"],
                silent=params["silent"],
                wo64=params["without_64"],
                mode = "train",
                hard_entire = True
            )["logits"]
            label_input_train_zeros = torch.zeros_like(label_input_train)
            if params["zeshel"]:
                
                train_tensor_data = TensorDataset(context_input_train, label_input_train_zeros, src_input_train, idxs_train, train_logits,)
            
            else:
                train_tensor_data = TensorDataset(context_input_train, label_input_train_zeros, src_input_train, idxs_train, train_logits)


            train_sampler = RandomSampler(train_tensor_data)

            train_dataloader = DataLoader(
            train_tensor_data, 
            sampler=train_sampler, 
            batch_size=params["train_batch_size"]
            )

        model.train()


        tr_loss = 0
        tr_loss_1 = 0
        tr_loss_2 = 0
        val_acc=0
        val_loss_sum=0
        val_loss_1_sum=0
        val_loss_2_sum=0
        results = None
        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")
            iter_valid = valid_dataloader
        print_interval=len(iter_)

        part = 0


        if params["architecture"]=="mlp":   
            model.eval()
            for step, batch in enumerate(iter_valid):
                batch = tuple(t.to(device) for t in batch)
                context_input = batch[0] 
                label_input = batch[1]
                idxs = batch[3]
                context_input, _ = modify(context_input, candidate_input_valid, params, world, idxs, mode = "valid", wo64 = params["without_64"])
                if params["dot_product"]:
                    val_loss, _, val_loss1, val_loss2 = reranker(context_input, label_input, context_length)
                    val_loss_1_sum += val_loss1.item()
                    val_loss_2_sum += val_loss2.item()
                else:
                    val_loss, _ = reranker(context_input, label_input, context_length)
                            # if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                val_loss_sum += val_loss.item()          
            logger.info(
                "epoch {} average validation loss: {}\n".format(
                    epoch_idx,
                    val_loss_sum / (len(iter_valid) * grad_acc_steps),
                )
            )
            if params["dot_product"]:
                wandb.log({
                "loss/val_loss": val_loss_sum / (len(iter_valid) * grad_acc_steps),
                "params/epoch": epoch_idx,
                "loss/val_label_loss": val_loss_1_sum / (len(iter_valid) * grad_acc_steps),
                "loss/val_dot_loss": val_loss_2_sum / (len(iter_valid) * grad_acc_steps),
                
                })
                val_loss_1_sum = 0
                val_loss_2_sum = 0

            else:
                wandb.log({
                "loss/val_loss": val_loss_sum / (len(iter_valid) * grad_acc_steps),
                "params/epoch": epoch_idx
                })
            val_loss_sum = 0
        if params["architecture"] != "raw_context_text" and params["architecture"] != "mlp_with_bert" and not params["hard_negative"]:
            logger.info("Evaluation on the training dataset")
            train_acc=evaluate(
                reranker,
                train_dataloader,
                candidate_input = candidate_input_train,
                device=device,
                logger=logger,
                context_length=context_length,
                zeshel=params["zeshel"],
                silent=params["silent"],
                wo64=params["without_64"]
            )
            wandb.log({
            "acc/train_acc": train_acc["normalized_accuracy"],
            "acc/train_acc(macro)": train_acc["macro_accuracy"],
            'mrr/train_mrr':train_acc["mrr"],
            "recall/train_recall":train_acc["recall"],
            "params/epoch": epoch_idx
            })
        elif params["train_size"] != -1:
            logger.info("Evaluation on the training dataset")
            train_acc=evaluate(
                reranker,
                train_dataloader,
                candidate_input = candidate_input_train,
                device=device,
                logger=logger,
                context_length=context_length,
                zeshel=params["zeshel"],
                silent=params["silent"],
                wo64=params["without_64"]
            )
            wandb.log({
            "acc/train_acc": train_acc["normalized_accuracy"],
            "acc/train_acc(macro)": train_acc["macro_accuracy"],
            'mrr/train_mrr':train_acc["mrr"],
            "recall/train_recall":train_acc["recall"],
            "params/epoch": epoch_idx
            })
        logger.info("Evaluation on the development dataset")
        val_acc=evaluate(
            reranker,
            valid_dataloader,
            candidate_input = candidate_input_valid,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
            mode = "valid",
            wo64=params["without_64"]
        )
        wandb.log({
            "acc/val_acc": val_acc["normalized_accuracy"],
            "acc/val_acc(macro)": val_acc["macro_accuracy"],
            "params/epoch": epoch_idx
            })
        logger.info("Evaluation on the development dataset (entire set)")
        val_acc_recall=evaluate(
                reranker,
                valid_dataloader,
                candidate_input = candidate_input_valid,
                device=device,
                logger=logger,
                context_length=context_length,
                zeshel=params["zeshel"],
                silent=params["silent"],
                mode = "valid",
                wo64=params["without_64"],
                recall_k = 64,
                hard_entire = True

            )
        wandb.log({
                "params/epoch":epoch_idx,
                "entire_instances/val_acc":val_acc_recall["normalized_accuracy"],
                "entire_instances/val_acc(macro)":val_acc_recall["macro_accuracy"],
                "entire_instances/val_recall":val_acc_recall["recall"],
            })
        if (val_acc["normalized_accuracy"]<=best_score):
            trigger_times+=1
            if (trigger_times>patience):
                logger.info("Early stopping by Patience")
                # if params["save"]:
                    # reranker.save_model(epoch_output_folder_path)
                nested_break=True
                break
        else:
            trigger_times=0
        print("trigger_times", trigger_times)

        last_acc=val_acc["normalized_accuracy"]

        ls = [best_score, val_acc["normalized_accuracy"]]
        ls_macro = [best_score_macro, val_acc["macro_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_score_macro = ls_macro[np.argmax(ls_macro)]
        best_epoch_idx = li[np.argmax(ls)]
        wandb.log({"acc/best_val_acc": best_score,
        "acc/best_val_acc(macro)": best_score_macro,
        "acc/best_epoch_idx": best_epoch_idx})

        # if (step + 1) % grad_acc_steps == 0:

        if params["without_64"]==False:
            wandb.log({         
                "params/learning_rate":  optimizer.param_groups[0]['lr'],
                "acc/train_acc": train_acc["normalized_accuracy"],
                "acc/train_acc(macro)": train_acc["macro_accuracy"],
                "acc_64/train_acc_64": train_acc["normalized_accuracy_64"],
                "acc_not64/train_acc_not64": train_acc["normalized_accuracy_not64"],
                "acc/val_acc":val_acc["normalized_accuracy"],
                "acc/val_acc(macro)":val_acc["macro_accuracy"],
                "acc_64/val_acc_64":val_acc["normalized_accuracy_64"],
                "acc_not64/val_acc_not64":val_acc["normalized_accuracy_not64"],
                'mrr/train_mrr':train_acc["mrr"],
                'acc_64/train_mrr_64':train_acc["mrr_64"],
                'mrr/val_mrr':val_acc['mrr'],
                'acc_64/val_mrr_64':val_acc['mrr_64'],
                "params/trigger_times":trigger_times,
                "recall/train_recall":train_acc["recall"],
                "recall/val_recall":val_acc["recall"],
                "params/epoch":epoch_idx
                })
        else:
            wandb.log({         
                "params/learning_rate":  optimizer.param_groups[0]['lr'],
                'mrr/val_mrr':val_acc['mrr'],
                "params/trigger_times":trigger_times,
                "recall/val_recall":val_acc["recall"],
                "params/epoch":epoch_idx
            })
        logger.info("\n")
        model.train()
        if nested_break==True:
            break

    ## Evaluation on test set
    test_split = params["test_split"]
    for i in range(test_split):
        if test_split == 1:
            fname = os.path.join(params["data_path"], "test_{}.t7".format(params["architecture"]))
        else:
            fname = os.path.join(params["data_path"], "test_{}_{}.t7".format(params["architecture"], i))
        if not os.path.isfile(fname):
            if params["architecture"] == "mlp":
                if valid_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("special_tokens"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("special_tokens", i))
            elif params["architecture"] == "special_tokens":
                if valid_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "som":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("mlp_with_som"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("mlp_with_som", i))
            elif params["architecture"] == "dot_product":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_extension":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("mlp"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("mlp", i))
            elif params["architecture"] == "extend_single":
                if train_split == 1:
                    fname = os.path.join(params["data_path"], "test_{}.t7".format("extend_multi"))
                else:
                    fname = os.path.join(params["data_path"], "test_{}_{}.t7".format("extend_multi", i))
        if i == 0:
            test_data = torch.load(fname)
            context_input_test = test_data["context_vecs"][:params["test_size"]]
            candidate_input_test = test_data["candidate_vecs"]
            idxs_test = test_data["indexes"][:params["test_size"]]
            label_input_test = test_data["labels"][:params["test_size"]]

        else:
            test_data = torch.load(fname)
            context_input_test = torch.cat((context_input_test, test_data["context_vecs"]), dim = 0)
            label_input_test = torch.cat((label_input_test, test_data["labels"]), dim = 0)
            idxs_test = test_data["indexes"][:params["test_size"]]
 
    
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    if params["zeshel"]:
        src_input_test = test_data['worlds'][:params["test_size"]]
    if params["eval_cands"] is not None:
        mask_eval_cands = label_input_test < params["eval_cands"]
        context_input_test = context_input_test[mask_eval_cands]
        idxs_test = idxs_test[mask_eval_cands] 
        label_input_test = idxs_test[mask_eval_cands]
        src_input_test = src_input_test[mask_eval_cands]
    if params["top_k"]<100 and not params["architecture"] in oom_architecture:
        context_input_test, _ = modify(context_input_test, candidate_input_test, params, src_input_test, idxs_test, mode = "test", wo64 = params["without_64"])

    if params["zeshel"]:
        test_tensor_data = TensorDataset(context_input_test, label_input, src_input, idxs)
    else:
        test_tensor_data = TensorDataset(context_input_test, label_input)

    test_sampler = RandomSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )

    logger.info("Evaluation on the test dataset")
    test_acc=evaluate(
        reranker,
        test_dataloader,
        candidate_input = candidate_input_test,
        device=device,
        logger=logger,
        context_length=context_length,
        zeshel=params["zeshel"],
        silent=params["silent"],
        wo64=params["without_64"]
    )
    if params["resume"]==True:
        wandb.init(project=params["wandb"], config=parser, resume="must", id=params["run_id"])
    else:
        wandb.init(project=params["wandb"], config=parser)
    if params["without_64"]==False:
        wandb.log({         
                "acc/test_acc":test_acc["normalized_accuracy"],
                "acc/test_acc(macro)":test_acc["macro_accuracy"],
                "wo64/test_acc_64":test_acc["normalized_accuracy_64"],
                "wo64/test_acc_not64":test_acc["normalized_accuracy_not64"],
                "recall/test_recall":test_acc["recall"],
                })
    else:
        wandb.log({         
            "acc/test_acc":test_acc["normalized_accuracy"],
            "recall/test_recall":test_acc["recall"],
            })

    # save the best model in the parent_dir
    # if (params["save"]):
    #     logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    #     params["path_to_model"] = os.path.join(
    #         model_output_path, "epoch_{}".format(best_epoch_idx)
    #     )
    #     torch.save(model.state_dict(), epoch_output_folder_path)
    # each_file_path_and_gen_time = []

    # for each_file_name in os.listdir(folder_path):
    #     each_file_path = folder_path + each_file_name
    #     each_file_gen_time = os.path.getctime(each_file_path)
    #     each_file_path_and_gen_time.append(
    #         (each_file_path, each_file_gen_time)
    #     )
    # most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
    # second_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[1]
    # for each_file_name in os.listdir(folder_path):
    #     each_file_path = folder_path + each_file_name
    #     if (each_file_path != most_recent_file and each_file_path != second_recent_file) and each_file_name[0]=="e":
    #         os.remove(each_file_path)


    # print("\rval_acc is {}".format(val_acc['mrr']))
    # print("\r")

    print("run_id:{}".format(run.id))

    
if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)