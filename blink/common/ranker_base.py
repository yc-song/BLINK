# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask, data_type="context"):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [M_s] and [ENT] token (context: token_id=1, candidate: token_id=2)
        if (data_type =="context"):
            m_s_index=(token_ids==1).nonzero(as_tuple=True)[1]
        elif (data_type=="candidate"):
            m_s_index=(token_ids==3).nonzero(as_tuple=True)[1]
        else:
            raise ValueError
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings=output_bert[range(output_bert.shape[0]),m_s_index]

        cls_token=output_bert[:,0,:]
        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings
        return result, cls_token

