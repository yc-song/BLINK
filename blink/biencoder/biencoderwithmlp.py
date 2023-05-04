# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

def load_biencoderwithMLP(params):
    # Init model
    biencoder = BiEncoderRankerwithMLP(params)
    return biencoder


class BiEncoderModulewithMLP(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModulewithMLP, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],

        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        cls_ctxt=None
        if token_idx_ctxt is not None:
            embedding_ctxt, cls_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        cls_cands=None
        if token_idx_cands is not None:
            embedding_cands, cls_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands, data_type="candidate"
            )
        return embedding_ctxt, embedding_cands, cls_ctxt, cls_cands


class BiEncoderRankerwithMLP(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRankerwithMLP, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.dropout=nn.Dropout(0.1)
        # init model
        self.build_model()
        self.mlp_model = MlpModule(params)
        self.mlp_model.to(self.device)
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModulewithMLP(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(''
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _, _, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands, _, cls_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach(), cls_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=False,
        cand_encs=None,  # pre-computed candidate encoding.
        cls_cands=None
    ):
        print(f"special token ids : {self.tokenizer.all_special_ids}")
        print(f"special tokens : {self.tokenizer.all_special_tokens}")

        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _, cls_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to rompute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            scores=cls_ctxt.mm(cls_cands.t()) 
            return scores, embedding_ctxt

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands, _, cls_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        input = torch.cat((embedding_ctxt, embedding_cands), dim = 1).to(torch.device('cuda'))

        scores = self.mlp_model(input)
        return scores, embedding_ctxt

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None):
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores[0].size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores[0], target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class MlpModule(nn.Module):
    def __init__(self, params):
        super(MlpModule, self).__init__()
        self.params=params
        self.input_size=768*2
        if params["bert_model"]=="bert-base-cased":
            self.input_size = 768*2
        self.fc=nn.Linear(self.input_size, self.input_size)
        self.fc2=nn.Linear(self.input_size, 1)
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        if params["act_fn"] == "softplus":
            self.act_fn = nn.Softplus()
        if params["act_fn"] == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif params["act_fn"] == "tanh":
            self.act_fn = nn.Tanh()
        self.current_dim = self.input_size
        if not self.params["decoder"]:
            if self.params["dim_red"]:
                for i in range(self.params["layers"]):
                    self.layers.append(nn.Linear(self.current_dim, self.params["dim_red"]))
                    self.current_dim = self.params["dim_red"]
            else: 
                for i in range(self.params["layers"]):
                    self.layers.append(nn.Linear(self.current_dim, self.current_dim))
        else:
            self.layers.append(nn.Linear(int(self.current_dim), int(self.params["dim_red"])))
            self.current_dim = self.params["dim_red"]
            for i in range(self.params["layers"]-1):
                self.layers.append(nn.Linear(int(self.current_dim), int(self.current_dim/2)))
                self.current_dim /= 2

        self.layers.append(nn.Linear(int(self.current_dim), 1))

        ## projection과 linear mapping 구분
        # relu나 sigmoid나 tanh
    def forward(self, input):
        for i, layer in enumerate(self.layers[:-1]):
            input = self.act_fn(layer(self.dropout(input)))
        input = self.layers[-1](self.dropout(input))
        return input