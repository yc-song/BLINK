# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import wandb
import pickle
import torch
import json
from torch import optim
import sys
import io
import random
import time
import numpy as np
import sys
sys.path.append('.')
from multiprocessing.pool import ThreadPool
import math
from tqdm import tqdm, trange
from collections import OrderedDict
import shutil
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BertTokenizer
WEIGHTS_NAME = "pytorch_model.bin"

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.biencoderwithmlp import BiEncoderRankerwithMLP, load_biencoderwithMLP

import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def modify(context_input, candidate_input, params, world, idxs, max_seq_length=256):
    # device = torch.device(
    #         "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
    #     )
    # # get values of candidates first
    # # candidate_input = candidate_input.to(device)
    # candidate_input = candidate_input[idxs.cpu()].to(device)
    # top_k=params["top_k"]
    # context_input = context_input.to(device)
    # context_input=context_input.unsqueeze(dim=1).expand(-1,top_k,-1)
    # new_input=torch.cat((context_input,candidate_input),dim=2)
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input[idxs.cpu()]
    candidate_input = candidate_input.tolist()

    for i in tqdm(range(len(context_input))):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)
        
    return torch.LongTensor(new_input)

def keep_recent_file(model_path, max_files, best_epoch_idx):
    best_epoch_idx = str(best_epoch_idx)
    files = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
    files = ["{}/{}".format(model_path, file) for file in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    if len(files) > max_files:
        del_list = files[0:max(len(files)-max_files, 0)]
        for del_file in del_list:
            if not del_file.endswith(best_epoch_idx):
                print("removed:", del_file)
                shutil.rmtree(del_file)

def find_most_recent_file(
    folder_path
):
    each_file_path_and_gen_time = []
    ## Getting newest file
    for each_file_name in os.listdir(folder_path):
        each_file_path = folder_path + each_file_name
        if not os.path.isfile(each_file_path):
            each_file_gen_time = os.path.getctime(each_file_path)
            each_file_path_and_gen_time.append(
                (each_file_path, each_file_gen_time)
            )
    most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0] + "/pytorch_model.bin"
    print(most_recent_file)

    return most_recent_file
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input, _ = batch

        candidate_input = context_input[:,:,params["max_context_length"]:]
        context_input = context_input[:,:,:params["max_context_length"]]
        context_input = context_input.reshape(context_input.size(0)*context_input.size(1), context_input.size(2))
        candidate_input = candidate_input.reshape(candidate_input.size(0)*candidate_input.size(1), candidate_input.size(2))
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)
        logits = logits[0].detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        # label_ids = torch.LongTensor(
        #         torch.arange(params["eval_batch_size"])
        # ).numpy()

        tmp_eval_accuracy, _ = utils.accuracy(logits, label_input)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results

def select_random_samples(N, batch_size, k):
    # N: number of entities
    # batch_size: # of context instances
    # k: # of hard negatives
    indices = torch.randperm(N)[:batch_size*k]
    indices = indices.reshape(batch_size, k)
    
    offsets = torch.arange(0, batch_size*k, k).reshape(batch_size, 1)
    offsets = offsets.expand(batch_size, k)
    
    random_tensor = (indices + offsets) % N
    
    return random_tensor # (N, k ) shaped tensor

def get_optimizer(model, params, architecture = None):
    if architecture is None:
        return get_bert_optimizer(
            [model],
            params["type_optimization"],
            params["bert_lr"],
            params["learning_rate"],
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
        print('The following parameters will be optimized WITH MLP optimizer')
        print(parameters_mlp_names[:5])
        optimizer_grouped_parameters = [
        {'params': parameters_mlp, 'weight_decay': 0.01}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=params["weight_decay"], lr=params["learning_rate"])
        return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = (math.ceil(len_train_data / batch_size / grad_acc)) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    if params["resume"]: 
        run = wandb.init(project=params["wandb"], config=parser, resume="must", id=params["run_id"])
    else:
        run = wandb.init(project="BLINK-biencoder")

    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

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

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    if params["type_cands"] != "random":
        train_samples = utils.read_dataset("train", params["data_path"])
        logger.info("Read %d train samples." % len(train_samples))
        train_data, train_tensor_data = data.process_mention_data(
            train_samples,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
        )
        if params["shuffle"]:
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )
        # Load Hard Negs.
    else:
        fname = os.path.join(params["candidate_path"], "train_hard.t7")
        if params["cpu"]:
            hard_data = torch.load(fname, map_location="cpu")
        else:
            hard_data = torch.load(fname)
        context_input = hard_data["context_vecs"].int()
        candidate_input_hard = hard_data["candidate_vecs"].int()
        label_input = hard_data["labels"]
        idxs = hard_data["indexes"]
        world = hard_data["worlds"]
        if params["debug"]:
            max_n = -1
            context_input = context_input[:max_n]
            # candidate_input = candidate_input[:max_n]
            label_input = label_input[:max_n]
            idxs = idxs[:max_n]

        if params["zeshel"]:
            src_input = hard_data["worlds"][:len(context_input)]
            train_tensor_data = TensorDataset(context_input, label_input, src_input, idxs)
        else:
            train_tensor_data = TensorDataset(context_input, label_input)
        train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, 
            sampler=train_sampler, 
            batch_size=params["train_batch_size"]
        )

    # Load eval data
    # TODO: reduce duplicated code here
    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    fname = os.path.join(params["candidate_path"], "valid.t7")
    if params["cpu"]:
        valid_data = torch.load(fname, map_location="cpu")
    else:
        valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"].int()
    candidate_input = valid_data["candidate_vecs"].int()
    label_input = valid_data["labels"]
    idxs = valid_data["indexes"]
    world = valid_data["worlds"]
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        # candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
        idxs = idxs[:max_n]

    context_input = modify(context_input, candidate_input, params, world, idxs)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )



    num_rands = int(params["train_batch_size"] * params["cands_ratio"])
    num_hards = params["train_batch_size"] - num_rands
    total_cands = candidate_input.size(0)




    number_of_samples_per_dataset = {}

    time_start = time.time()



    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    optimizer_mlp = None
    epoch_idx_global = -1
    if params["classification_head"] == "mlp":
        optimizer_mlp = get_optimizer(model, params, architecture = "mlp")
    if params["resume"]: 
        model_output_path="models/zeshel/biencoder/{}/".format(run.id)
        most_recent_file = find_most_recent_file(model_output_path)
        reranker, epoch_idx_global, optimizer = utils.load_model(most_recent_file, reranker, optimizer)
        optimizer.step()
        if optimizer_mlp is not None:
            optimizer_mlp.step()
        for i in range(math.ceil(len(train_tensor_data) / params["train_batch_size"] / params["gradient_accumulation_steps"]) * (epoch_idx_global+1)):
            scheduler.step()
    else:
        model_output_path="models/zeshel/biencoder/{}/".format(run.id)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )
    # evaluate before training
    results = evaluate(
        reranker,
        valid_dataloader,
        params,
        device=device,
        logger=logger,
        # context_length=context_length,
        # zeshel=params["zeshel"],
        # silent=params["silent"],
    )

    wandb.log({"accuracy": results["normalized_accuracy"], "epoch": epoch_idx_global})

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    print(optimizer)
    for epoch_idx in trange(epoch_idx_global+1, int(num_train_epochs), desc="Epoch"):
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path, -1, optimizer)
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, label_input, world, nn_idxs = batch
            ### modify required here
            if params["type_cands"] == "random":
                target_sets = nn_idxs[torch.arange(nn_idxs.size(0)), label_input].unsqueeze(-1) # (Batch_size,)
                # print(target_sets, target_sets.shape)
                target_input = candidate_input_hard[target_sets]
                # print(target_input, target_input.shape)
                candidate_sets = select_random_samples(candidate_input_hard.size(0), context_input.size(0), params["top_k"]-1) # (batch_size, top_k-1)
                # print(candidate_sets, candidate_sets.shape)
                candidate_input= candidate_input_hard[candidate_sets] #(batch_size, batch_size-1, context_len)
                # print(candidate_input, candidate_input.shape)
                candidate_input = torch.cat([target_input, candidate_input], dim = -2)
                # print(candidate_input, candidate_in/put.shape)
                context_input = context_input.unsqueeze(dim = 1).expand(-1, params["top_k"], -1)
                context_input = context_input.reshape(context_input.size(0)*context_input.size(1), context_input.size(2))
                candidate_input = candidate_input.reshape(candidate_input.size(0)*candidate_input.size(1), candidate_input.size(2))
            loss, _ = reranker(context_input, candidate_input)
            # for i, param in enumerate(list(reranker.named_parameters())):
            #     print(param)
            #     print(list(reranker.parameters())[i].grad)
            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                ) 
                wandb.log({"train_loss": tr_loss / (params["print_interval"] * grad_acc_steps), "epoch": epoch_idx, "learning_rate": optimizer.param_groups[0]['lr']})

                tr_loss = 0
            # print("***0****")
            # for n, p in model.named_parameters():
            #     if p.grad is not None:
            #         print(n)
            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                if optimizer_mlp is not None:
                    optimizer_mlp.step()
                scheduler.step()
                optimizer.zero_grad()
                # print("***1****")
                # for n, p in model.named_parameters():
                #     if "embeddings" in n:
                #         print(p.requires_grad)
                #         print(p.grad)
                if optimizer_mlp is not None:
                    # print("***2****")
                    optimizer_mlp.zero_grad()
                    # for n, p in model.named_parameters():
                    #     if p.grad is not None:
                    #         print(n)


            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results = evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                wandb.log({"accuracy": results["normalized_accuracy"], "epoch":epoch_idx})
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path, epoch_idx, optimizer)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        wandb.log({"accuracy": results["normalized_accuracy"], "epoch":epoch_idx,\
        "best_accuracy": best_score, "learning_rate": optimizer.param_groups[0]['lr']})
        logger.info("***** Removing old models *****")
        keep_recent_file(model_output_path[:-1], 1, best_epoch_idx)

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker.load_model(params["path_to_model"])
    utils.save_model(reranker.model, tokenizer, model_output_path, epoch_idx, optimizer)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
