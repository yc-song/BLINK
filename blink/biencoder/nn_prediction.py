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
import numpy as np
import os
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from blink.biencoder.biencoder import to_bert_input, BiEncoderRanker
from transformers.models.bert.modeling_bert import BertModel
def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    cand_cls_list,
    top_k=64,
    is_zeshel=False,
    save_predictions=False,
    params=None,
    cand_encode_late_interaction = None,
    split = 0,
    candidate_pool_ids = None,
    mention_ids = None,
    candidate_encode_list_frozen = None
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)
    if params['freeze_bert']:
        bert_path = params["path_to_model"]
        params["path_to_model"] = None
        reranker_frozen = BiEncoderRanker(params)
        params["path_to_model"] = bert_path
    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    nn_scores = []
    nn_idxs = []
    cands = []
    stats = {}
    if params["mode"] == "valid":
        src_minus = 8
    elif params["mode"] == "test":
        src_minus = 12
    else:
        src_minus = 0
    if is_zeshel:
        world_size = len(WORLDS)
        cumulative_idx = [0]
        cand_encode_late_interaction_tensor = None
        cand_encode_list_tensor = None
        candidate_pool_tensor = None
        # Stacking cand_encode_list and candidate_pool to one tensor
        for i in range(1, len(cand_encode_list)):
            cumulative_idx.append(cumulative_idx[i-1]+cand_encode_list[i-1+src_minus].size(0))
                # World 0, 1, 2... have been concatenated in tensor [tensors of world 0, tensors of world 1, tensors of world 2, ...]
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
        if params["classification_head"] == "dot" or params["classification_head"] == "extend_multi":
            scores, embedding_ctxt, embedding_late_interaction_ctxt = reranker.score_candidate(
                context_input, 
                None, 
                cand_encs=cand_encode_list[src],
                cls_cands=cand_cls_list[src],
            ) #scores: (batch_size, each_world_size)
            if top_k == -1:
                values = scores
                # indicies = torch.arange(scores.size(1)).expand(scores.size(0), -1)
                values, indicies = scores.sort(descending = True)  
            else:
                values, indicies = scores.topk(top_k)

            old_src = src


        for i in range(context_input.size(0)):
            if step == 0:
                MENTION_SIZE = context_input.size(0)
            oid += 1
            if params["classification_head"] == "dot" or params["classification_head"] == "extend_multi":
                inds = indicies[i]
                value = values[i]
                src = srcs[i].item()
                if src != old_src:
                    # not the same domain, need to re-do
                    new_scores, _, _ = reranker.score_candidate(
                        context_input[[i]], 
                        None,
                        cand_encs=cand_encode_list[src].to(device),
                        cls_cands=cand_cls_list[src].to(device)
                    )
                    if top_k == -1:
                        value = new_scores
                        # inds = torch.arange(new_scores.size(1)).unsqueeze(0)
                        value, inds = new_scores.sort(descending = True)  

                    else:
                        value, inds = new_scores.topk(top_k)
                    inds = inds[0]
                    value = value[0]
            else:
                src = srcs[i].item()
                new_scores, _, _ = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    cand_encs=cand_encode_list[src].to(device),
                    cls_cands=cand_cls_list[src].to(device),
                    embedding_late_interaction_cands = cand_encode_late_interaction[src]
                )
                if top_k == -1:
                    value = new_scores
                    # inds = torch.arange(new_scores.size(1)).unsqueeze(0)
                    value, inds = new_scores.sort(descending = True)  
                else:
                    value, inds = new_scores.topk(top_k)
                inds = inds[0]
                value = value[0]  
            pointer = -1
            pointer_temp = (inds == label_ids[i].item()).nonzero(as_tuple=True)[0]
            if pointer_temp.size(0) != 0:
                pointer = pointer_temp.item()
            stats[src].add(pointer)

            if params["save_topk_nce"]:
                mention_id = mention_ids[i + MENTION_SIZE*step]
                candidate = {}
                candidate['mention_id'] = mention_id
                # candidate['mention_id'] = hex(mention_id).upper()[2:].rjust(16, "0")
                all_entities = np.array(candidate_pool_ids[srcs[i].item()])
                candidate['candidates'] = all_entities[inds.cpu()].tolist()
                candidate['scores'] = value.tolist()
                candidate['labels'] = pointer
                cands.append(candidate)
                # cur_candidates = candidate_pool[srcs[i].item()][inds]
            # else:
                # cur_candidates = cand_encode_list[srcs[i].item()][inds]#(64,1024)
            if pointer == -1:
                continue 
            if save_predictions:
                if params["freeze_bert"]:
                   nn_context.append(reranker_frozen.encode_context(context_input[i].unsqueeze(0)).squeeze(-2).cpu().tolist())
                elif params["architecture"] == "raw_context_text" or params["architecture"] == "mlp_with_bert":
                    nn_context.append(context_input[i].cpu().tolist())#(1024)
                elif params["architecture"] == "mlp_with_som" or params["architecture"] == "extend_multi":
                    nn_context.append(embedding_late_interaction_ctxt[i].cpu().tolist())#(1024)
                
                    # if type(nn_context) is list:
                    #     nn_context = embedding_late_interaction_ctxt
                    # else:
                    #     nn_context = torch.cat([nn_context, embedding_late_interaction_ctxt], dim=0)
                else:
                    nn_context.append(embedding_ctxt[i].cpu().tolist())#(1024)
                nn_idxs.append([x + cumulative_idx[srcs[i].item()-src_minus] for x in inds.tolist()])
                nn_scores.append(value.tolist())
                nn_labels.append(pointer)
                nn_worlds.append(srcs[i].item()-src_minus)

            # if pointer == -1:
            #     # pointer = j + 1

            #     continue

            else: 
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
    if params["save_topk_nce"]:
        cands_dir = os.path.join(
                params['output_path'],
                "top%d_candidates" % params['top_k'],
            )
        if not os.path.exists(cands_dir):
            os.makedirs(cands_dir)
        with open(os.path.join(cands_dir, 'candidates_%s.json' % params["mode"]), 'w') as f:
            for item in cands:
                f.write('%s\n' % json.dumps(item))
        return
    else:
        if not save_predictions:
            return output
        nn_context = torch.FloatTensor(nn_context) # (10000,1024)
        nn_labels = torch.LongTensor(nn_labels)
        max_length = max(len(sublist) for sublist in nn_idxs)
        # As inference is carried out on each world. Dimension mismatch occurs b/w instances.
        # Assign largest long to mismatching indices 
        nn_idxs = [sublist + [torch.iinfo(torch.long).max] * (max_length - len(sublist)) for sublist in nn_idxs]
        nn_idxs = torch.LongTensor(nn_idxs)
        # For the same reason, assign -inf to mismatching scores to prevent them from drawn at sampling
        nn_scores = [sublist + [-float('inf')] * (max_length - len(sublist)) for sublist in nn_scores]
        nn_scores = torch.FloatTensor(nn_scores)
        nn_data = {
            'context_vecs': nn_context,
            'labels': nn_labels,
            'nn_scores': nn_scores,
            'indexes': nn_idxs,
            'cumulative_index': cumulative_idx
        }
        # print("candidate", nn_data["candidate_vecs"])
        if save_predictions:
            print("context shape", nn_data["context_vecs"].shape)
            print("score shape", nn_data["nn_scores"].shape)
            print("index shape", nn_data["indexes"].shape)
            print("labels", nn_data["labels"].shape)
        if split == 0:
            if params['freeze_bert']:
                for i in range(len(cand_encode_list)):
                    if i == 0:
                        cand_encode_list_tensor = candidate_encode_list_frozen[i+src_minus]
                    else:
                        cand_encode_list_tensor = torch.cat([cand_encode_list_tensor, candidate_encode_list_frozen[i+src_minus]])
                nn_data["candidate_vecs"] = cand_encode_list_tensor.to(device)        

                 
            elif params["architecture"] == "raw_context_text" or params["architecture"] == "mlp_with_bert":
                for i in range(len(cand_encode_list)):
                    if i == 0:
                        candidate_pool_tensor = candidate_pool[i+src_minus]
                    else:
                        candidate_pool_tensor = torch.cat([candidate_pool_tensor, candidate_pool[i+src_minus]])

                nn_data["candidate_vecs"] = candidate_pool_tensor.to(device)
            elif params["architecture"] == "mlp_with_som":
                for i in range(len(cand_encode_list)):
                    if i == 0:
                        cand_encode_late_interaction_tensor = cand_encode_late_interaction[i+src_minus]
                    else:
                        cand_encode_late_interaction_tensor = torch.cat([cand_encode_late_interaction_tensor, cand_encode_late_interaction[i+src_minus]], dim=0)            
                nn_data["candidate_vecs"] = cand_encode_late_interaction_tensor
            
            else:
                for i in range(len(cand_encode_list)):
                    if i == 0:
                        cand_encode_list_tensor = cand_encode_list[i+src_minus]
                    else:
                        cand_encode_list_tensor = torch.cat([cand_encode_list_tensor, cand_encode_list[i+src_minus]])
                nn_data["candidate_vecs"] = cand_encode_list_tensor.to(device)        

            print("candidate", nn_data["candidate_vecs"].shape)

        if is_zeshel:
            nn_data["worlds"] = torch.LongTensor(nn_worlds)
        print("worlds", nn_data["worlds"].shape)
        
    return nn_data

