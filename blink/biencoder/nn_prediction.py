# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    cand_cls_list,
    cand_cls_list,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
    params=None
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    nn_scores = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]
        cand_cls_list=[cand_cls_list]
        cand_cls_list=[cand_cls_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        if is_zeshel:
            context_input, _, srcs, label_ids = batch
        else:
            context_input, _, label_ids = batch
            srcs = torch.tensor([0] * context_input.size(0), device=device)
        if len(batch)==4:
            context_input, _, srcs, label_ids = batch
            src = srcs[0].item()
        elif len(batch) == 3:
            context_input, _, label_ids = batch
            src = 0
        cand_encode_list[src] = cand_encode_list[src].to(device)
        cand_cls_list[src] = cand_cls_list[src].to(device)

        cand_cls_list[src] = cand_cls_list[src].to(device)

        scores, embedding_ctxt = reranker.score_candidate(
            context_input, 
            None, 
            cand_encs=cand_encode_list[src],
            cls_cands=cand_cls_list[src]
        ) #scores: (batch_size, each_world_size)
        values, indicies = scores.topk(top_k)
        old_src = src
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]
            value = values[i]
            if srcs[i] != old_src:
                src = srcs[i].item()
                # not the same domain, need to re-do
                new_scores, _ = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    cand_encs=cand_encode_list[src].to(device),
                    cls_cands=cand_cls_list[src].to(device)
                )
                value, inds = new_scores.topk(top_k)
                inds = inds[0]
                value = values[0]
            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)
# Speical tokens
            # if params["mode"] != "train":
            #     if pointer == -1:
            #         pointer = j + 1
            #         cand_encode_list[srcs[i].item()]=cand_encode_list[srcs[i].item()].to(device)
            #         cur_candidates = cand_encode_list[srcs[i].item()][inds]#(64,1024)
            #         cur_candidates=torch.cat((cur_candidates,cand_encode_list[srcs[i].item()][label_ids[i]]), dim=0) #(65,1024)
            #         if params["architecture"] == "raw_context_text":
            #             nn_context.append(context_input[i].cpu().tolist())#(1024)
            #         else:
            #             nn_context.append(embedding_ctxt[i].cpu().tolist())#(1024)

            #     else:
            #         cand_encode_list[srcs[i].item()]=cand_encode_list[srcs[i].item()].to(device)
            #         cur_candidates = cand_encode_list[srcs[i].item()][inds]#(64,1024)
            #         if params["architecture"] == "raw_context_text":
            #             nn_context.append(context_input[i].cpu().tolist())#(1024)
            #         else:
            #             nn_context.append(embedding_ctxt[i].cpu().tolist())#(1024)
            #         if params["bert_model"]=="bert-large-uncased":
            #             cur_candidates=torch.cat((cur_candidates,torch.rand((1, 1024), device=device)), dim=0) #(65,1024)
            #         elif params["bert_model"]=="bert-base-cased":
            #             cur_candidates=torch.cat((cur_candidates,torch.rand((1, 768), device=device)), dim=0) #(65,1024)
                
            #     # nn_context.append(embedding_ctxt[i].cpu().tolist())#(1024)

            if pointer == -1:
                continue
            cand_encode_list[srcs[i].item()] = cand_encode_list[srcs[i].item()].to(device)
            cur_candidates = cand_encode_list[srcs[i].item()][inds]#(64,1024)
            if params["architecture"] == "raw_context_text":
                nn_context.append(context_input[i].cpu().tolist())#(1024)
            else:
                nn_context.append(embedding_ctxt[i].cpu().tolist())#(1024)
            nn_scores.append(value.tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)
            # if pointer == -1:
            #     # pointer = j + 1

            #     continue

            if not save_predictions:
                continue

            # # add examples in new_data
            # cand_encode_list[srcs[i].item()]=cand_encode_list[srcs[i].item()].to(device)
            # cur_candidates = cand_encode_list[srcs[i].item()][inds]
            # nn_context.append(embedding_ctxt[i].cpu().tolist())
            # nn_candidates.append(cur_candidates.cpu().tolist())
            # nn_labels.append(pointer)
            # nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.FloatTensor(nn_context) # (10000,1024)
    nn_candidates = torch.FloatTensor(nn_candidates) # (10000,64,1024) -> (10000,65,1024)
    nn_labels = torch.LongTensor(nn_labels)
    nn_scores = torch.FloatTensor(nn_scores)

    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
        'nn_scores': nn_scores
    }
    print("context shape", nn_context.shape)
    print("candidate shape", nn_candidates.shape)
    print("score shape", nn_scores.shape)
    
    print("labels", nn_labels.shape)
    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data

