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
logger = None

def modify(context_input, candidate_input, params):
    top_k=params["top_k"]
    # print("context", context_input.shape)
    # print("candidate", candidate_input.shape)
    ## context_input shape: (Size ,1024) e.g.  (10000, 1024)
    ## candidate_input shape: (Size, 65, 1024) e.g. (10000, 65, 1024)
    context_input=context_input.unsqueeze(dim=1).expand(-1,top_k+1,-1)
    if params["architecture"]=="mlp" or params["architecture"]=="special_token":
        new_input=torch.stack((context_input,candidate_input),dim=2) # shape: (Size, 65, 2, 1024) e.g. (10000, 65, 2 , 1024)
    elif params["architecture"]== "raw_context_text":
        new_input=torch.cat((context_input,candidate_input),dim=2)
    # print(new_input.shape)
    # print(new_input.shape)
    return new_input


def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True, train=True):
    reranker.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    eval_mrr = 0.0
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
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)
        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        eval_recall += utils.recall(logits, label_ids)
        # print("recall", eval_recall)
        eval_mrr+=utils.mrr(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
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
    logger.info("Recall@4: %.5f" % eval_recall)


    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    results["mrr"]=eval_mrr
    results["recall"]=eval_recall
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    gc.collect()
    nested_break=False
    wandb.init(project=params["wandb"], config=parser)
    config = wandb.config
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    if params["architecture"]=="mlp":
        reranker= MlpModel(params)
    else:
        reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

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

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    fname = os.path.join(params["data_path"], "test.t7")

    # fname2 = os.path.join(params["data_path"], "train2.t7")
    train_data = torch.load(fname)
    torch.set_printoptions(threshold=10_000)
    context_input = train_data["context_vecs"][:params["train_size"]]
    # print("context shape:", context_input.shape)
    # print("context:", context_input)

    # print("train context", context_input.shape) 
    candidate_input = train_data["candidate_vecs"][:params["train_size"]]
    # print("candidate shape:", candidate_input.shape)
    # print("candidate:", candidate_input)
    label_input = train_data["labels"][:params["train_size"]]
    # print(label_input)
    bi_train_mrr=torch.mean(1/(label_input+1)).item()

    logger.info("train bi-encoder mrr: {}".format(bi_train_mrr))


    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, params)
    if params["zeshel"]:
        src_input = train_data['worlds'][:params["train_size"]]
        train_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        train_tensor_data = TensorDataset(context_input, label_input)
    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )
    fname = os.path.join(params["data_path"], "valid.t7")

    valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"][:params["valid_size"]]
    candidate_input = valid_data["candidate_vecs"][:params["valid_size"]]
    label_input = valid_data["labels"][:params["valid_size"]]
    bi_val_mrr=torch.mean(1/(label_input+1)).item()
    logger.info("valid bi-encoder mrr: {}".format(bi_val_mrr))


    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    # print("valid context", context_input.shape)
    # print("valid candidate", candidate_input.shape)
    context_input = modify(context_input, candidate_input, params)
    # print("valid modify", context_input.shape)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:params["valid_size"]]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    # evaluate before training
    results = evaluate(
        reranker,
        valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        zeshel=params["zeshel"],
        silent=params["silent"],
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )
    # optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"])
    # if params["optimizer"]=="Adam":
    #     optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    #     print(optimizer)

    # elif params["optimizer"]=="SGD":
    #     optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"])
    # elif params["optimizer"]=="RMSprop":
        # optimizer = optim.RMSprop(model.parameters(), lr=params["learning_rate"])
 
    if params["architecture"] == "mlp":
        if params["optimizer"]=="Adam":
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        elif params["optimizer"]=="AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])

        elif params["optimizer"]=="SGD":
            optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params["learning_rate"])
        elif params["optimizer"]=="RMSprop":
            optimizer = optim.RMSprop(model.parameters(),lr=params["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["scheduler_gamma"])

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.9, min_lr=0.000001)
    elif params["architecture"]=="special_token" or params["architecture"]=="raw_context_text":
        optimizer = get_optimizer(model, params)
        scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    model.train()

    best_epoch_idx = -1
    best_score = -1
    #Early stopping variables
    patience=params["patience"]
    last_acc=1
    trigger_times=-1
    print_interval=params["print_interval"]
    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        val_acc=0
        val_loss_sum=0
        results = None
        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")
            iter_valid = valid_dataloader
        print_interval=len(iter_)

        part = 0
        model.train()
        # print("\n opt", optimizer)
        print("learning rate", scheduler.get_last_lr())


        for step, batch in enumerate(iter_):
            torch.autograd.set_detect_anomaly(True)
            model.train()
            batch = tuple(t.to(device) for t in batch)
            # print(batch)
            # print(batch[0].shape)
            # print(batch[1].shape)
            context_input = batch[0] 
            label_input = batch[1]
            loss, _ = reranker(context_input, label_input, context_length)
            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()
            # if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
            if (step+1)%(print_interval*grad_acc_steps)==0 :
                
                logger.info(
                    "Step {} - epoch {} average training loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (print_interval * grad_acc_steps),
                    )
                )
                for step, batch in enumerate(iter_valid):
                    model.eval()
                    batch = tuple(t.to(device) for t in batch)
                    context_input = batch[0] 
                    label_input = batch[1]
                    val_loss, _ = reranker(context_input, label_input, context_length)
                                # if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                    val_loss_sum += val_loss.item()
                    
                    if (step + 1) == (len(iter_)):
                        
                        logger.info(
                            "Step {} - epoch {} average validation loss: {}\n".format(
                                step,
                                epoch_idx,
                                val_loss_sum / (print_interval * grad_acc_steps),
                            )
                        )
                # logger.info("Evaluation on the training dataset")
                # train_acc=evaluate(
                #     reranker,
                #     train_dataloader,
                #     device=device,
                #     logger=logger,
                #     context_length=context_length,
                #     zeshel=params["zeshel"],
                #     silent=params["silent"],
                # )
                
                # logger.info("Evaluation on the development dataset")
                # val_acc=evaluate(
                #     reranker,
                #     valid_dataloader,
                #     device=device,
                #     logger=logger,
                #     context_length=context_length,
                #     zeshel=params["zeshel"],
                #     silent=params["silent"],
                #     train=False
                # )
                # if (val_acc["normalized_accuracy"]<=last_acc):
                #     trigger_times+=1
                #     print("trigger_times", trigger_times)
                #     if (trigger_times>patience):
                #         print("Early stopping")
                #         # if params["save"]:
                #             # reranker.save_model(epoch_output_folder_path)
                #         nested_break=True
                #         break
                # else:
                #     print("valid accuracy got better")
                #     trigger_times=0
                # last_acc=val_acc["normalized_accuracy"]
                
        



            # if (step+1)%(params["print_interval"]*grad_acc_steps) == 0:
            #     logger.info("Evaluation on the training dataset")
            #     train_acc=evaluate(
            #         reranker,
            #         train_dataloader,
            #         device=device,
            #         logger=logger,
            #         context_length=context_length,
            #         zeshel=params["zeshel"],
            #         silent=params["silent"],
            #     )
                
            #     logger.info("Evaluation on the development dataset")
            #     val_acc=evaluate(
            #         reranker,
            #         valid_dataloader,
            #         device=device,
            #         logger=logger,
            #         context_length=context_length,
            #         zeshel=params["zeshel"],
            #         silent=params["silent"],
            #         train=False
            #     )
                
            #     wandb.log({"train_acc":train_acc['normalized_accuracy'], "val_acc":val_acc['normalized_accuracy']})

            #     # for step, batch in enumerate(iter_valid):
            #     #     model.eval()
            #     #     context_input = batch[0] 
            #     #     label_input = batch[1]
            #     #     loss, _ = reranker(context_input, label_input, context_length)
            #     #     if grad_acc_steps > 1:
            #     #         loss = loss / grad_acc_steps

            #     #     val_loss += loss.item()

            #     #     if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
            #     #         logger.info(
            #     #             "Step {} - epoch {} average val loss: {}\n".format(
            #     #                 step,
            #     #                 epoch_idx,
            #     #                 val_loss / (params["print_interval"] * grad_acc_steps),
            #     #             )
            #     #         )

            #     if (val_acc["normalized_accuracy"]<last_acc):
            #         trigger_times+=1
            #         print("trigger_times", trigger_times)
            #         if (trigger_times>patience):
            #             print("Early stopping")
            #             # if params["save"]:
            #                 # reranker.save_model(epoch_output_folder_path)
            #             nested_break=True
            #             break
            #     else:
            #         print("valid accuracy got better")
            #         trigger_times=0
            #     last_acc=val_acc["normalized_accuracy"]

            #     # logger.info("***** Saving fine - tuned model *****")
            #     # if (params["save"]):
            #     #     epoch_output_folder_path = os.path.join(
            #     #         model_output_path, "epoch_{}_{}".format(epoch_idx, part)
            #     #     )
            #     #     part += 1
            #     #     # utils.save_model(model, tokenizer, epoch_output_folder_path)
            #     #     torch.save(model.state_dict(), epoch_output_folder_path)
            #     # reranker.save_model(epoch_output_folder_path)
            #     model.train()
            #     logger.info("\n")
    
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        # utils.save_model(model, tokenizer, epoch_output_folder_path)
        # reranker.save_model(epoch_output_folder_path)
        logger.info("Evaluation on the training dataset")
        train_acc=evaluate(
            reranker,
            train_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
        )

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        
        logger.info("Evaluation on the development dataset")
        val_acc=evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
            train=False
        )
        if (val_acc["normalized_accuracy"]<=last_acc):
            trigger_times+=1
            if (trigger_times>patience):
                print("Early stopping")
                # if params["save"]:
                    # reranker.save_model(epoch_output_folder_path)
                nested_break=True
                break
        else:
            trigger_times=0
        print("trigger_times", trigger_times)

        last_acc=val_acc["normalized_accuracy"]

        ls = [best_score, val_acc["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]

        loss.backward()

        # if (step + 1) % grad_acc_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), params["max_grad_norm"]
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        wandb.log({         
            "learning_rate": scheduler.get_last_lr(),
            "train_bi-encoder_mrr":bi_train_mrr,
            "valid_bi-encoder_mrr":bi_val_mrr,
            "train_loss":tr_loss / (len(iter_) * grad_acc_steps),
            "val_loss":val_loss_sum / (len(iter_) * grad_acc_steps),
            "train_acc": train_acc["normalized_accuracy"],
            "val_acc":val_acc["normalized_accuracy"],
            'train_mrr':train_acc["mrr"],
            'val_mrr':val_acc['mrr'],
            "trigger_times":trigger_times,
            "train_recall":train_acc["recall"],
            "val_recall":val_acc["recall"],
            "epoch":epoch_idx
            })
        val_loss_sum = 0
        tr_loss = 0
        logger.info("\n")

        if nested_break==True:
            break

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))


    # save the best model in the parent_dir
    if (params["save"]):
        logger.info("Best performance in epoch: {}".format(best_epoch_idx))
        params["path_to_model"] = os.path.join(
            model_output_path, "epoch_{}".format(best_epoch_idx)
        )
        torch.save(model.state_dict(), epoch_output_folder_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)
