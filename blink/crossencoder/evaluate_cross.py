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
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
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
import deepspeed
# from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt
logger = None

def modify(context_input, candidate_input, params, world, idxs, mode = "train", wo64 = True):
    device = torch.device('cuda')
    # get values of candidates first
    # candidate_input = candidate_input.to(device)
    print(idxs.shape)
    print(candidate_input.shape)
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
            new_input = torch.stack((context_input, candidate_input), dim = 2)
        else:
            context_input=context_input.unsqueeze(dim=1).expand(-1,top_k,-1).to(device)
            if params["architecture"] =="mlp" or params["architecture"]=="special_token" or params["architecture"]=="mlp_with_bert" or params["architecture"]=="baseline":
                new_input=torch.stack((context_input,candidate_input),dim=2) # shape: (Size, 65, 2, 1024) e.g. (10000, 65, 2 , 1024)
            elif params["architecture"] == "raw_context_text":
                new_input=torch.cat((context_input,candidate_input),dim=2)
        return new_input
        


def evaluate(reranker, eval_dataloader, device, logger, context_length, candidate_input, zeshel=False, silent=True, train=True, input = None, wo64 = True):
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

    all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        if params["top_k"]>100 or params["architecture"] == "mlp_with_som" or params["architecture"] == "som":
            world = batch[2]
            idxs = batch[3]
            context_input = modify(context_input, candidate_input, params, world, idxs, mode = "train", wo64 = params["without_64"])

        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length, evaluate = True)
        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        
        eval_recall += utils.recall(logits, label_ids)
        # print("recall", eval_recall)
        eval_mrr+=utils.mrr(logits, label_ids, train=train, input = context_input)
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
        all_logits.extend(logits)

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


    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    results["mrr"]=eval_mrr
    if params["without_64"] == False:
        results["mrr_64"]=eval_mrr_64
        results["normalized_accuracy_64"]=normalized_eval_accuracy_64
        results["normalized_accuracy_not64"]=normalized_eval_accuracy_not64
    results["recall"]=eval_recall
    return results



def main(params):
    gc.collect()
    nested_break=False
    # wandb.init(project=params["wandb"], config=parser, resume="must", id=<original_sweeps_run_id>)
    epoch_idx_global = 0
    previous_step = 0
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
        if i == 0:
            train_data = torch.load(fname,  map_location=torch.device('cpu'))
            # train_data = torch.load(fname)
            context_input = train_data["context_vecs"][:params["train_size"]]
            candidate_input_train = train_data["candidate_vecs"]
            idxs = train_data["indexes"][:params["train_size"]]
            label_input = train_data["labels"][:params["train_size"]]
            bi_encoder_score = train_data["nn_scores"][:params["train_size"]]
            if params["zeshel"]:
                src_input = train_data['worlds'][:params["train_size"]]
        else:
            # fname2 = os.path.join(params["data_path"], "train2.t7")
            train_data = torch.load(fname)
            context_input = torch.cat((context_input, train_data["context_vecs"][:params["train_size"]]), dim = 0)
            idxs = torch.cat((idxs, train_data["indexes"][:params["train_size"]]), dim = 0)
            label_input = torch.cat((label_input, train_data["labels"][:params["train_size"]]), dim = 0)
            bi_encoder_score = torch.cat((bi_encoder_score, train_data["nn_scores"][:params["train_size"]]), dim = 0)
            if params["zeshel"]:
                src_input = torch.cat((src_input, train_data['worlds'][:params["train_size"]]), dim = 0)
    # Init model
    if params["architecture"]=="mlp":
        reranker= MlpModel(params)
    elif params["architecture"] == "mlp_with_bert":
        reranker = MlpwithBiEncoderRanker(params)
    elif params["architecture"] == "mlp_with_som":
        reranker = MlpwithSOMRanker(params)
    elif params["architecture"] == "som":
        reranker = SOMRanker(params)
    else:
        reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    model.eval()


    # if you want to resume the training, set "resume" true and specify "run_id"
    if len(params["run_id"]) == 8:
        folder_path="models/zeshel/crossencoder/{}/{}/".format(params["architecture"], params["run_id"])
        each_file_path_and_gen_time = []
        ## Getting newest file
        for each_file_name in os.listdir(folder_path):
            if each_file_name[0]=="e":
                each_file_path = folder_path + each_file_name
                each_file_gen_time = os.path.getctime(each_file_path)
                each_file_path_and_gen_time.append(
                    (each_file_path, each_file_gen_time)
                )
        most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
        run = wandb.init(project=params["wandb"], config=parser, resume="must", id=params["run_id"])
        print("file loaded:", most_recent_file)
        checkpoint = torch.load(most_recent_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_idx_global = checkpoint['epoch']
        previous_step = checkpoint['step']
    else:
        run = wandb.init(project=params["wandb"], config=parser)
    config = wandb.config
    model_output_path = params["output_path"]+params["architecture"]+"/"+run.id
    logger = utils.get_logger(params["output_path"])
    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    train_loss_sum=0
    val_loss_sum=0
    # print(label_input)
    bi_train_mrr=torch.mean(1/(label_input+1)).item()
    bi_train_accuracy = torch.mean((label_input == 0).float()).item()
    bi_train_recall_4 = torch.mean((label_input <= 3).float()).item()


    logger.info("train bi-encoder mrr: {}".format(bi_train_mrr))
    wandb.log({
    "mrr/train_bi-encoder_mrr":bi_train_mrr,
    "acc/train_bi-encoder_accuracy": bi_train_accuracy,
    "recall/train_bi-encoder_recall@4": bi_train_recall_4,


    })

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    if params["top_k"]<100 and params["architecture"] != "mlp_with_som" and params["architecture"] != "som":
        context_input = modify(context_input, candidate_input_train, params, src_input, idxs, mode = "train", wo64 = params["without_64"])
    if params["zeshel"]:
        train_tensor_data = TensorDataset(context_input, label_input, src_input, idxs, bi_encoder_score)
    else:
        train_tensor_data = TensorDataset(context_input, label_input, src_input, idxs, bi_encoder_score)


    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )
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
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp_with_som"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp_with_som", i))

        if i == 0:
            valid_data = torch.load(fname)
            context_input = valid_data["context_vecs"][:params["valid_size"]]
            candidate_input_valid = valid_data["candidate_vecs"]
            idxs = valid_data["indexes"][:params["valid_size"]]
            label_input = valid_data["labels"][:params["valid_size"]]
            bi_encoder_score = valid_data["nn_scores"][:params["valid_size"]]
            if params["zeshel"]:
                src_input = valid_data['worlds'][:params["valid_size"]]
        else:
            # fname2 = os.path.join(params["data_path"], "train2.t7")
            valid_data = torch.load(fname)
            context_input = torch.cat((context_input, valid_data["context_vecs"][:params["valid_size"]]), dim = 0)
            idxs = torch.cat((idxs, valid_data["indexes"][:params["valid_size"]]), dim = 0)
            label_input = torch.cat((label_input, valid_data["labels"][:params["valid_size"]]), dim = 0)
            bi_encoder_score = torch.cat((bi_encoder_score, valid_data["nn_scores"][:params["valid_size"]]), dim = 0)
            if params["zeshel"]:
                src_input = valid_data['worlds'][:params["valid_size"]]
    bi_val_mrr=torch.mean(1/(label_input+1)).item()
    bi_val_accuracy = torch.mean((label_input == 0).float()).item()
    bi_val_recall_4 = torch.mean((label_input <= 3).float()).item()
    wandb.log({
    "mrr/val_bi-encoder_mrr":bi_val_mrr,
    "acc/val_bi-encoder_accuracy": bi_val_accuracy,
    "recall/val_bi-encoder_recall@4": bi_val_recall_4,
    })


    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    # print("valid context", context_input.shape)
    # print("valid candidate", candidate_input.shape)
    if params["top_k"]<100 and params["architecture"] != "mlp_with_som" and params["architecture"] != "som":
        context_input = modify(context_input, candidate_input_valid, params, src_input, idxs, mode = "valid", wo64=params["without_64"])
    # print("valid modify", context_input.shape)
    valid_tensor_data = TensorDataset(context_input, label_input, src_input, idxs)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )
    iter_train = tqdm(train_dataloader, desc="Batch")
    iter_valid = valid_dataloader
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
    'mrr/train_mrr':train_acc["mrr"],
    "recall/train_recall":train_acc["recall"],
    })

    # logger.info("Loss on the training dataset")
    # for step, batch in enumerate(iter_train):
    #     batch = tuple(t.to(device) for t in batch)
    #     context_input = batch[0] 
    #     label_input = batch[1]
    #     if params["top_k"]>100 or params["architecture"] == "mlp_with_som" or params["architecture"] == "som":
    #         world = batch[2]
    #         idxs = batch[4]
    #         context_input = modify(context_input, candidate_input_valid, params, world, idxs, mode = "valid", wo64 = params["without_64"])
    #     train_loss, _ = reranker(context_input, label_input, context_length)
    #                 # if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
    #     train_loss_sum += train_loss.item()
                
                    
    # logger.info(
    #     "training loss: {}\n".format(
    #         train_loss_sum / (len(iter_train) * grad_acc_steps),
    #     )
    # )
    
    train_loss_sum = 0
    wandb.log({
    "loss/train_loss": train_loss_sum / (len(iter_valid) * grad_acc_steps),
    })
    val_loss_sum = 0


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
        train=False,
        wo64=params["without_64"]
    )
    wandb.log({
        "acc/val_acc": val_acc["normalized_accuracy"],
        'mrr/val_mrr':val_acc['mrr'],
        "recall/val_recall":val_acc["recall"],
        })

    number_of_samples_per_dataset = {}

    # logger.info("Loss on the validation dataset")

    # for step, batch in enumerate(iter_valid):
    #     batch = tuple(t.to(device) for t in batch)
    #     context_input = batch[0] 
    #     label_input = batch[1]
    #     if params["top_k"]>100 or params["architecture"] == "mlp_with_som" or params["architecture"] == "som":
    #         world = batch[2]
    #         idxs = batch[4]
    #         context_input = modify(context_input, candidate_input_valid, params, world, idxs, mode = "valid", wo64 = params["without_64"])
    #     val_loss, _ = reranker(context_input, label_input, context_length)
    #                 # if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
    #     val_loss_sum += val_loss.item()
                
                    
    logger.info(
        "validation loss: {}\n".format(
            val_loss_sum / (len(iter_valid) * grad_acc_steps),
        )
    )
    wandb.log({
    "loss/val_loss": val_loss_sum / (len(iter_valid) * grad_acc_steps),
    })
    val_loss_sum = 0



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
                    fname = os.path.join(params["data_path"], "train_{}.t7".format("mlp_with_som"))
                else:
                    fname = os.path.join(params["data_path"], "train_{}_{}.t7".format("mlp_with_som", i))
        if i == 0:
            test_data = torch.load(fname)
            context_input = test_data["context_vecs"][:params["test_size"]]
            candidate_input_test = test_data["candidate_vecs"]
            idxs = test_data["indexes"][:params["test_size"]]
            label_input = test_data["labels"][:params["test_size"]]

        else:
            test_data = torch.load(fname)
            context_input = torch.cat((context_input, test_data["context_vecs"]), dim = 0)
            label_input = torch.cat((label_input, test_data["labels"]), dim = 0)

    
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    if params["zeshel"]:
        src_input = test_data['worlds'][:params["test_size"]]
    if params["top_k"]<100 and params["architecture"] != "mlp_with_som" and params["architecture"] != "som":
        context_input = modify(context_input, candidate_input_test, params, src_input, idxs, mode = "test", wo64 = params["without_64"])
    if params["zeshel"]:
        test_tensor_data = TensorDataset(context_input, label_input, src_input, idxs)
    else:
        test_tensor_data = TensorDataset(context_input, label_input, src_input, idxs)
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
        train=False,
        wo64=params["without_64"]
    )
    if params["resume"]==True:
        wandb.init(project=params["wandb"], config=parser, resume="must", id=params["run_id"])
    else:
        wandb.init(project=params["wandb"], config=parser)
    if params["without_64"]==False:
        wandb.log({         
                "acc/test_acc":test_acc["normalized_accuracy"],
                "wo64/test_acc_64":test_acc["normalized_accuracy_64"],
                "wo64/test_acc_not64":test_acc["normalized_accuracy_not64"],
                "recall/test_recall":test_acc["recall"],
                })
    else:
        wandb.log({         
            "acc/test_acc":test_acc["normalized_accuracy"],
            "recall/test_recall":test_acc["recall"],
            })


    
if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)