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
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from collections import OrderedDict
from transformers.models.bert.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.crossencoder.mlp import MlpModule, MlpModel
def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(BiEncoderModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        if params["classification_head"] == "mlp":
            self.mlplayer = MlpModule(params).to(self.device)
        if params["anncur"]:
            config = BertConfig.from_pretrained(params["bert_model"], output_hidden_states=True)
            ctxt_bert = BertModel.from_pretrained(params["bert_model"], config=config)
            cand_bert = BertModel.from_pretrained(params['bert_model'], config=config)
        else:
            ctxt_bert = BertModel.from_pretrained(params["bert_model"])
            cand_bert = BertModel.from_pretrained(params['bert_model'])
        ctxt_bert.resize_token_embeddings(len(tokenizer))
        cand_bert.resize_token_embeddings(len(tokenizer))
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            tokenizer = tokenizer,
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            params = params
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            tokenizer = tokenizer,
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            params = params
        )
        self.config = ctxt_bert.config
        self.params = params
        if params["classification_head"] == "extend_multi" or params["classification_head"] == "extend_multi_full":
            self.n_heads = params["n_heads"]
            self.num_layers = params["num_layers"]
            self.embed_dim = 768
            self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, batch_first = True).to(self.device)
            self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers).to(self.device)
            self.linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
            self.sep_embedding = torch.normal(0, 1, size = (1, self.embed_dim)).to(self.device)

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt = None,
        mask_ctxt = None,
        token_idx_cands = None,
        segment_idx_cands = None,
        mask_cands = None,
    ):
        # if token_idx_cands is None:
        #     print("*** For Evaluation Purpose ***")
        #     token_idx_cands = torch.randint(1, 3, (64, 128)).to(self.device)
        #     segment_idx_ctxt = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        #     segment_idx_cands = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        #     mask_ctxt = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        #     mask_cands = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        embedding_ctxt = None
        cls_ctxt=None
        embedding_late_interaction_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt, cls_ctxt, embedding_late_interaction_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, params = self.params
            )
        embedding_cands = None
        cls_cands=None
        embedding_late_interaction_cands = None
        if token_idx_cands is not None:
            embedding_cands, cls_cands, embedding_late_interaction_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands, data_type="candidate", params = self.params
            )
        return embedding_ctxt, embedding_cands, cls_ctxt, cls_cands, embedding_late_interaction_ctxt, embedding_late_interaction_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer

        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        if params["anncur"]:
            self.tokenizer = BertTokenizer.from_pretrained(
			"bert-base-uncased", do_lower_case=True
		)
        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        self.NULL_IDX = self.tokenizer.pad_token_id
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path, cpu = params["cpu"])
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)



    def load_model(self, fname, cpu=False):
        if self.params["anncur"]:
            if cpu:
                state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")["state_dict"]
            else:
                state_dict = torch.load(fname)["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.input_encoder'):
                    name = k.replace('model.input_encoder', 'context_encoder')
                elif k.startswith('model.label_encoder'):
                    name = k.replace('model.label_encoder', 'cand_encoder')
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict = False)

        else:
            if not cpu:
                state_dict = torch.load(fname, map_location="cpu")
            else:
                state_dict = torch.load(fname)
            new_state_dict = OrderedDict()
            try:
                for k, v in state_dict["state_dict"].items():
                    if not k.startswith('model'):
                        name = "model."+k
                        new_state_dict[name] = v
                    else:
                        name = k
                        new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            except KeyError:
                for k, v in state_dict.items():
                    if not k.startswith('model'):
                        name = "model.module."+k
                        new_state_dict[name] = v
                    else:
                        name = k
                        new_state_dict[name] = v
                self.load_state_dict(new_state_dict, strict=False)

    def build_model(self):
        self.model = BiEncoderModule(self.params, self.tokenizer)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)

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
        _, embedding_cands, _, cls_cands, _, embedding_late_interaction = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach(), cls_cands.cpu().detach(), embedding_late_interaction.cpu().detach()
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
        cls_cands=None,
        embedding_late_interaction_cands = None,
    ):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _, cls_ctxt, _, embedding_late_interaction_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:

            if self.params["classification_head"] == "som":
                # ctxt_reshaped = embedding_late_interaction_ctxt.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), \
                # embedding_late_interaction_ctxt.size(2)).to(device)
                # cands_reshaped = embedding_late_interaction_cands.reshape(embedding_late_interaction_cands.size(0)* embedding_late_interaction_cands.size(1)\
                # , embedding_late_interaction_cands.size(2)).to(device)
                # output = torch.mm(ctxt_reshaped, cands_reshaped.t())
                # output = output.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), embedding_late_interaction_cands.size(0) , \
                # embedding_late_interaction_cands.size(1)).max(-1)[0]
                # output = output.t().reshape(embedding_late_interaction_cands.size(0)*embedding_late_interaction_ctxt.size(0), embedding_late_interaction_ctxt.size(1))
                # scores = torch.sum(output, dim = -1).reshape(embedding_late_interaction_cands.size(0), embedding_late_interaction_ctxt.size(0)).t()
                # print(scores.shape)
                batch_size = 32
                scores = None
                for i in range(0, embedding_late_interaction_cands.size(0), batch_size):
                    batch = embedding_late_interaction_cands[i:i+batch_size]                
                    ctxt_reshaped = embedding_late_interaction_ctxt.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), \
                    embedding_late_interaction_ctxt.size(2)).cpu()
                    cands_reshaped = batch.reshape(batch.size(0)* batch.size(1)\
                    , batch.size(2))
                    output = torch.mm(ctxt_reshaped, cands_reshaped.t())
                    output = output.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), batch.size(0) , \
                    batch.size(1)).max(-1)[0]
                    output = output.t().reshape(batch.size(0)*embedding_late_interaction_ctxt.size(0), embedding_late_interaction_ctxt.size(1))
                    if scores is None:
                        scores = torch.sum(output, dim = -1).reshape(batch.size(0), embedding_late_interaction_ctxt.size(0)).t().cpu()
                    else:
                        scores = torch.cat([scores, torch.sum(output, dim = -1).reshape(batch.size(0), embedding_late_interaction_ctxt.size(0)).t().cpu()], dim = -1) 
            elif self.params["classification_head"] == "extend_multi":
                pass
            elif self.params["classification_head"] == "mlp":
                num_ctxt = cls_ctxt.size(0)
                num_cands = cls_cands.size(0)
                cls_ctxt = cls_ctxt.unsqueeze(1) # (64, 1, 768)
                cls_cands = cls_cands.unsqueeze(0) # (1, # of cands, 768)
                # Expand tensors using broadcasting
                cls_ctxt = cls_ctxt.expand(-1, num_cands, -1)
                cls_cands = cls_cands.expand(num_ctxt, -1, -1)
                # Reshape tensors to (64*64, 2)
                cls_ctxt = cls_ctxt.reshape(num_ctxt*num_cands, 768)
                cls_cands = cls_cands.reshape(num_ctxt*num_cands, 768)

                # Stack the tensors along the second dimension
                input = torch.stack([cls_ctxt, cls_cands], dim=1)
                scores = self.model.module.mlplayer(input)
                scores = scores.reshape(num_ctxt, num_cands)
            elif self.params["classification_head"] == "dot":
                scores=cls_ctxt.mm(cls_cands.t())
            return scores, embedding_ctxt, embedding_late_interaction_ctxt
        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )

        _, embedding_cands, _, cls_cands, _,  embedding_late_interaction_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        # if self.params["with_mlp"]:
        #     embedding_ctxt = embedding_ctxt.unsqueeze(dim=1).expand(-1,embedding_ctxt.size(0),-1)
        #     input=torch.stack((embedding_ctxt,embedding_cands),dim=2)
        #     input = torch.cat((embedding_ctxt.unsqueeze(dim = 1).unsqueeze(dim = 2), embedding_cands.unsqueeze(dim = 1).unsqueeze(dim = 2)), dim = 2)
        #     scores = self.mlp_model(input)
        if random_negs:
            batch_size = cls_ctxt.size(0)
            if self.params["classification_head"]=="mlp":
                cls_ctxt = cls_ctxt.unsqueeze(1) # (64, 1, 768)
                cls_cands = cls_cands.unsqueeze(0) # (1, 64, 768)
                # Expand tensors using broadcasting
                cls_ctxt = cls_ctxt.expand(batch_size,batch_size, 768)
                cls_cands = cls_cands.expand(batch_size, batch_size, 768)
                # Reshape tensors to (64*64, 2)
                cls_ctxt = cls_ctxt.reshape(batch_size*batch_size, 768)
                cls_cands = cls_cands.reshape(batch_size*batch_size, 768)

                # Stack the tensors along the second dimension
                input = torch.stack([cls_ctxt, cls_cands], dim=1)
                scores = self.model.module.mlplayer(input)
                scores = scores.reshape(batch_size, batch_size)
            elif self.params["classification_head"]=="extend_multi":
                # ToDo: add separate token
                cls_ctxt = cls_ctxt.unsqueeze(-2)
                cls_cands = cls_cands.expand(batch_size, -1, -1)
                input = torch.cat([cls_ctxt, cls_cands], dim = -2)
                attention_result = self.model.module.transformerencoder(input)
                scores = self.model.module.linearhead(attention_result[:,-batch_size:,:])
                scores = scores.squeeze(-1)
            elif self.params["classification_head"]=="extend_multi_full":
                # ToDo: add separate token
                #embedding_late_interaction: (64, 128, 768)
                cls_cands = cls_cands.expand(batch_size, -1, -1) # (64, 64, 768)
                sep_embedding = self.model.module.sep_embedding.expand(batch_size, -1, -1)
                input = torch.cat((embedding_late_interaction_ctxt, sep_embedding, cls_cands), dim = -2) # (64, 128 + 64, 768)
                attention_result = self.model.module.transformerencoder(input)
                scores = self.model.module.linearhead(attention_result[:,-batch_size:,:])
                scores = scores.squeeze(-1)
            elif self.params["classification_head"] == "som":
                scores = None
                ctxt_reshaped = embedding_late_interaction_ctxt.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), \
                embedding_late_interaction_ctxt.size(2))
                cands_reshaped = embedding_late_interaction_cands.reshape(embedding_late_interaction_cands.size(0)* embedding_late_interaction_cands.size(1)\
                , embedding_late_interaction_cands.size(2))
                output = torch.mm(ctxt_reshaped, cands_reshaped.t())
                output = output.reshape(embedding_late_interaction_ctxt.size(0)*embedding_late_interaction_ctxt.size(1), embedding_late_interaction_cands.size(0) , \
                embedding_late_interaction_cands.size(1)).max(-1)[0]
                output = output.t().reshape(embedding_late_interaction_cands.size(0)*embedding_late_interaction_ctxt.size(0), embedding_late_interaction_ctxt.size(1))
                if scores is None:
                    scores = torch.sum(output, dim = -1).reshape(embedding_late_interaction_cands.size(0), embedding_late_interaction_ctxt.size(0)).t()
                else:
                    scores = torch.cat([scores, torch.sum(output, dim = -1).reshape(embedding_late_interaction_cands.size(0), embedding_late_interaction_ctxt.size(0)).t()], dim = -1) 

                # # (64, 1, 128, 768)
                # embedding_late_interaction_ctxt = embedding_late_interaction_ctxt.unsqueeze(1)
                # # (1, 64, 128, 768)
                # embedding_late_interaction_cands = embedding_late_interaction_cands.unsqueeze(0)
                # # expand to (64, 64, 128, 768)
                # embedding_late_interaction_ctxt = embedding_late_interaction_ctxt.expand(batch_size,batch_size, 128, 768)
                # embedding_late_interaction_cands = embedding_late_interaction_cands.expand(batch_size,batch_size, 128, 768)
                # # reshape to (64*64, 128, 768)
                # embedding_late_interaction_ctxt = embedding_late_interaction_ctxt.reshape(batch_size*batch_size, 128, 768)
                # embedding_late_interaction_cands = embedding_late_interaction_cands.reshape(batch_size*batch_size, 128, 768)
                # output = torch.bmm(embedding_late_interaction_ctxt, embedding_late_interaction_cands.transpose(1, 2))
                # scores = torch.max(output, dim = -1)[0]
                # scores = torch.sum(scores, dim = -1)
                # scores = scores.reshape(batch_size, batch_size)
            # train on random negatives
            else:
                # train on hard negatives 
                scores=cls_ctxt.mm(cls_cands.t())
        else:
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(cls_ctxt, cls_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
        return scores, embedding_ctxt, embedding_late_interaction_ctxt

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