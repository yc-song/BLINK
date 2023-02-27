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

from collections import OrderedDict
from tqdm import tqdm
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from blink.crossencoder.mlp import (
    MlpModel,
)
from pytorch_transformers.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

from blink.common.ranker_base_cross import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

def load_crossencoder(params):
    # Init model
    crossencoder = CrossEncoderRanker(params)
    return crossencoder

class CrossEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(CrossEncoderModule, self).__init__()
        model_path = params["bert_model"]
        if params.get("roberta"):
            encoder_model = RobertaModel.from_pretrained(model_path)
        elif params["architecture"] == "special_token":
            encoder_model = SpecialTokenBertModel.from_pretrained(model_path)
        elif params["architecture"] == "raw_context_text":
            encoder_model = RawTextBertModel.from_pretrained(model_path)
            encoder_model.params=params
        else:
            encoder_model = BertModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        self.encoder = BertEncoder(
            encoder_model,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = self.encoder.bert_model.config

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
    ):
        embedding_ctxt = self.encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        return embedding_ctxt.squeeze(-1)


class CrossEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(CrossEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        if params.get("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(params["bert_model"],)
        else:
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
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token

        # init model
        self.build_model()
        if params["path_to_model"] is not None:
            self.load_model(params["path_to_model"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def save(self, output_dir):
        self.save_model(output_dir)
        self.tokenizer.save_vocabulary(output_dir)

    def build_model(self):
        self.model = CrossEncoderModule(self.params, self.tokenizer)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def score_candidate(self, text_vecs, context_len):
        # Encode contexts first
        num_cand = text_vecs.size(1) # 65
        if self.params["architecture"] == "special_token":
            text_vecs = text_vecs.view(-1, text_vecs.size(-2), text_vecs.size(-1))
        else: 
            text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            text_vecs, self.NULL_IDX, context_len,
        )
        embedding_ctxt = self.model(token_idx_ctxt, segment_idx_ctxt, mask_ctxt,)
        # (2080)이 나와야 함
        # check if add_linear true
        return embedding_ctxt.view(-1, num_cand)

    def forward(self, input_idx, label_input, context_len, evaluate = False):
        scores = self.score_candidate(input_idx, context_len)
        loss = F.cross_entropy(scores, label_input, reduction="mean")
        return loss, scores


    def to_bert_input(self, token_idx, null_idx, segment_pos):
        """ token_idx is a 2D tensor int.
            return token_idx, segment_idx and mask
        """
        # token_idx shape: (2080, 2, 1024)
        segment_idx = token_idx * 0
        if segment_pos > 0:
            segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] != 0
        if self.params["architecture"]=="special_token":
            # mask dimension is supposed to be (2080, 2) rather than (2080, 2, 1024)
            mask=torch.ones((token_idx.size(0), token_idx.size(1)),dtype=torch.bool)
        else:
            mask = token_idx != null_idx

        # nullify elements in case self.NULL_IDX was not 0
        # token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask

class SpecialTokenBertModelwithSEPToken(BertModel):
    """
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(SpecialTokenBertModel, self).__init__(config)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(input_ids,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class SpecialTokenBertModel(BertModel):
    """
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(SpecialTokenBertModel, self).__init__(config)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        cls_tokens = torch.empty((input_ids.shape[0], 1), device = "cuda").fill_(101.)
        sep_tokens = torch.empty((input_ids.shape[0], 1), device = "cuda").fill_(102.)
        input_tokens = torch.cat((cls_tokens, sep_tokens), dim = 1)
        token_type_ids = torch.zeros_like(input_tokens)
        embedding_output = self.embeddings(input_tokens.int(), position_ids=position_ids, token_type_ids=token_type_ids.int())
        input_ids = torch.cat((torch.unsqueeze(embedding_output[:,1,:], dim = 1), torch.unsqueeze(input_ids[:,0,:], dim = 1), torch.unsqueeze(embedding_output[:,1,:], dim = 1), torch.unsqueeze(input_ids[:,1,:], dim = 1)), dim = 1)
        print("input_ids", input_ids.shape)

        attention_mask = torch.ones_like(input_ids[:,:,0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(input_ids,
                                       extended_attention_mask,
                                       head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class RawTextBertModel(BertModel):
    """
    Inputs:
    Context: Raw Text same as the BLINK paper
    Entity: [ENT] token from bi-encoder

    """
    def __init__(self, config):
        super(RawTextBertModel, self).__init__(config)
        self.params=None
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        torch.set_printoptions(threshold=10_000)
        max_context_length=self.params["max_context_length"]
        attention_mask=attention_mask[:,:max_context_length+1]
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        device = torch.device("cuda")
        np.set_printoptions(threshold=1000000)
        token_type_ids=torch.ones_like(token_type_ids[:,:max_context_length])
        # embedding_output = self.embeddings(input_ids[:,:max_context_length].int(), position_ids=position_ids, token_type_ids=token_type_ids[:,:max_context_length].int()) 
        embedding_output = self.embeddings(input_ids[:,:max_context_length].int(), position_ids=position_ids, token_type_ids=token_type_ids.int()) 
        embedding_output= torch.cat((embedding_output,input_ids[:,max_context_length:].unsqueeze(dim=1)), dim = 1)

        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)