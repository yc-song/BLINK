# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, tokenizer, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask, data_type="context", params = None):
        
        output = self.bert_model(
            input_ids = token_ids, attention_mask = attention_mask, token_type_ids = segment_ids
        )
        all_embeddings = output[0]
        cls_token = output[0][:,0,:]
        # get embedding of [M_s] and [ENT] token (context: token_id=1, candidate: token_id=2)
        # all_embeddings = output_bert
        # m_s_token_id = self.tokenizer.encode(ENT_START_TAG)[0]
        # ent_token_id = self.tokenizer.encode(ENT_TITLE_TAG)[0]
        # if (data_type =="context"):
        #     m_s_index=(token_ids==m_s_token_id).nonzero(as_tuple=True)[1]
        # elif (data_type=="candidate"):
        #     m_s_index=(token_ids==ent_token_id).nonzero(as_tuple=True)[1]
        # else:
        #     raise ValueError
        # if self.additional_linear is not None:
        #     embeddings = output_pooler
        # else:
        #     print("*********")
        #     print(output_bert)
        #     embeddings=output_bert[range(output_bert.shape[0]),m_s_index]
        # # mask = self.mask(token_ids).to(self.device)
        # # all_embeddings = output_bert*mask
        # # all_embeddings = torch.nn.functional.normalize(all_embeddings, p = 2, dim = 2)
        # cls_token=output_bert[:,0,:]
        # # in case of dimensionality reduction
        if self.additional_linear is not None:
            cls_token = self.additional_linear(self.dropout(cls_token))
        return cls_token, cls_token, all_embeddings
    
    def mask(self, token_ids, skiplist = []):
        pad_token_id = self.tokenizer.encode("[PAD]")[0]
        mask = [[(x not in skiplist) and (x != pad_token_id) for x in d] for d in token_ids.cpu().tolist()]
        return torch.tensor(mask).unsqueeze(dim = -1)

