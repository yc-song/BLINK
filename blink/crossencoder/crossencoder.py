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
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.adapters.context import ForwardContext
from transformers.adapters.composition import adjust_tensors_for_parallel
from collections import OrderedDict
from tqdm import tqdm
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
from transformers.adapters.models.bert.adapter_model import BertAdapterModel
from transformers.adapters import BertAdapterModel, AutoAdapterModel
# from transformers.models.bert.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertLayer
from blink.crossencoder.mlp import (
    MlpModel,
)
# from pytorch_transformers.modeling_roberta import (
#     RobertaConfig,
#     RobertaModel,
# )

from transformers.models.bert.tokenization_bert import BertTokenizer
# from pytorch_transformers.tokenization_roberta import RobertaTokenizer

# from blink.common.ranker_base_cross import BertEncoder, get_model_obj
from blink.common.ranker_base import BertEncoder

from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.crossencoder.mlp import MlpModule, MlpModel
from blink.biencoder.biencoder import BiEncoderModule, BiEncoderRanker

def load_crossencoder(params):
    # Init model
    crossencoder = CrossEncoderRanker(params)
    return crossencoder

class CrossEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(CrossEncoderModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        model_path = params["bert_model"]
        if params.get("roberta"):
            encoder_model = RobertaModel.from_pretrained(model_path)
        elif params["architecture"] == "special_token":
            encoder_model = SpecialTokenBertModel.from_pretrained(model_path)
        elif params["architecture"] == "raw_context_text":
            encoder_model = RawTextBertModel.from_pretrained(model_path)
            encoder_model.params=params
        elif params["architecture"] == "baseline":
            config = BertConfig.from_pretrained(model_path, output_hidden_states=True)
            encoder_model = BertModel.from_pretrained(model_path, config = config)
        elif params["architecture"] == "mlp_with_bert":
            encoder_model = BertModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        if params["architecture"] == "baseline":
            self.encoder = BertEncoder(
                encoder_model,
                params["out_dim"],
                tokenizer,
                layer_pulled=params["pull_from_layer"],
                add_linear=params["add_linear"],
            )
            self.config = self.encoder.bert_model.config

        else:
            self.encoder = BertEncoder(
                encoder_model,
                params["out_dim"],
                tokenizer,
                layer_pulled=params["pull_from_layer"],
                add_linear=params["add_linear"],
            )
            self.config = self.encoder.bert_model.config

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt = None, mask_ctxt = None,
    ):
        # if segment_idx_ctxt is None:
        #     print("*** For Evaluation Purpose ***")
        #     segment_idx_ctxt = torch.zeros(token_idx_ctxt.shape).to(self.device)
        # if mask_ctxt is None:
        #     print("*** For Evaluation Purpose ***")
        #     mask_ctxt = torch.zeros(token_idx_ctxt.shape).to(self.device)

        embedding_ctxt, _, _ = self.encoder(token_idx_ctxt.int(), segment_idx_ctxt.int(), mask_ctxt.int())
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
        if params["path_to_model"] is not None and params["anncur"] is None:
            print("load")
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
        return embedding_ctxt.reshape(-1, num_cand)

    def forward(self, input_idx, label_input, context_len, evaluate = False):
        scores = self.score_candidate(input_idx, context_len)
        # print(input_idx.shape)
        # print(scores.shape)
        # print(label_input.shape)
        loss = F.cross_entropy(scores, label_input, reduction="mean")
        print(loss)
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
class ExtendSingleModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(ExtendSingleModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )  
        self.params = params
        self.n_heads = params["n_heads"]
        self.embed_dim = 768
        self.num_layers = params["num_layers"]
        self.top_k = 64
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, batch_first = True)
        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers)
        self.attentionlayer = torch.nn.MultiheadAttention(self.embed_dim, self.n_heads, batch_first = True)
        self.classification_head = torch.nn.Linear(self.embed_dim, 1)

        # self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        # self.feedforward = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )
    def forward(self, input):
        # attention_result = self.attentionlayer(input, input, input)[0]
        # attention_result = self.layer_norm(input+attention_result)
        # linear_result = self.feedforward(attention_result)
        # attention_result = self.layer_norm(linear_result+attention_result)
        batched_input = input.reshape(-1, input.size(-2), input.size(-1))
        attention_result = self.transformerencoder(batched_input)
        scores = self.classification_head(attention_result[:,0,:])
        scores = scores.reshape(input.size(0), input.size(1))
        return scores.squeeze(-1)
class ExtendSingleRanker(torch.nn.Module):
    def __init__(self, params):
        super(ExtendSingleRanker, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        self.tokenizer = None   
        self.model = ExtendSingleModule(params, self.tokenizer)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, input, label_input, context_length, evaluate = False):
        scores = self.model(input)
        loss = self.criterion(scores, label_input)
        return loss, scores


class ExtendMultiModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(ExtendMultiModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )  
        self.params = params
        self.n_heads = params["n_heads"]
        self.embed_dim = 768
        self.num_layers = params["num_layers"]
        self.top_k = 64
        self.multiplicative_parameters = torch.nn.Parameter(torch.eye(self.embed_dim).unsqueeze(0)) # (1, 768, 768)
        self.mlplayer = MlpModule(self.params).to(self.device)
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, dim_feedforward=self.embed_dim*4, batch_first = True)
        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers)
        self.attentionlayer = torch.nn.MultiheadAttention(self.embed_dim, self.n_heads, batch_first = True)
        self.classification_head = torch.nn.Linear(self.embed_dim, 1)
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)

        # self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        # self.feedforward = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )
    def forward(self, input):
        # attention_result = self.attentionlayer(input, input, input)[0]
        # attention_result = self.layer_norm(input+attention_result)
        # linear_result = self.feedforward(attention_result)
        # attention_result = self.layer_norm(linear_result+attention_result)
        token_type_context = torch.zeros(input.size(0), 128).int().to(self.device)
        token_embedding_context = self.token_type_embeddings(token_type_context)
        token_type_candidate = torch.ones(input.size(0), input.size(1)-128).int().to(self.device)
        token_embedding_candidate = self.token_type_embeddings(token_type_candidate)
        token_embedding_type = torch.cat([token_embedding_context, token_embedding_candidate], dim = 1)
        input += token_embedding_type
        attention_result = self.transformerencoder(input)
        if self.params["classification_head"] == "dot":
            if self.params["pooling"] == "sum":
                context_input = torch.sum(attention_result[:,:-self.top_k,:], dim = 1).unsqueeze(1)
            elif self.params["pooling"] == "cls":
                context_input = attention_result[:,0,:].unsqueeze(1)
            candidate_input = attention_result[:,-self.top_k:,:]
            context_input = context_input.expand(-1, self.top_k, -1).reshape(-1, 1, context_input.size(-1))
            candidate_input = candidate_input.reshape(-1, context_input.size(-1), 1)
            scores = torch.bmm(context_input, candidate_input)
            # scores = torch.bmm(context_input, self.multiplicative_parameters.expand(context_input.size(0), -1, -1))
            # scores = torch.bmm(scores, candidate_input)
            scores = scores.reshape(input.size(0), input.size(1)-128)
        elif self.params["classification_head"] == "linear":
            scores = self.classification_head(attention_result)[:,-self.top_k:,:]
        elif self.params["classification_head"] == "mlp":
            if self.params["pooling"] == "sum":
                context_input = torch.sum(attention_result[:,:-self.top_k,:], dim = 1).unsqueeze(1).expand(-1, self.params["top_k"], -1)
            elif self.params["pooling"] == "cls":
                context_input = attention_result[:,0,:].unsqueeze(1).expand(-1, self.params["top_k"], -1)
            candidate_input = attention_result[:,-self.top_k:,:]
            context_input = context_input.reshape(-1, context_input.size(-1))
            candidate_input = candidate_input.reshape(-1, context_input.size(-1))
            mlp_input = torch.stack((context_input, candidate_input), dim = 1)
            scores = self.mlplayer(mlp_input)
            scores = scores.reshape(input.size(0), input.size(1)-128)
        return scores.squeeze(-1)
class ExtendMultiRanker(torch.nn.Module):
    def __init__(self, params):
        super(ExtendMultiRanker, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.params = params
        self.n_gpu = torch.cuda.device_count()
        self.tokenizer = None   
        self.model = ExtendMultiModule(params, self.tokenizer)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, input, label_input, context_length, evaluate = False):
        scores = self.model(input)
        loss = self.criterion(scores, label_input)
        return loss, scores

class ExtendExtensionBertModule(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config,  add_pooling_layer=True):
        super(ExtendExtensionBertModule, self).__init__(config)
        self.n_heads=8
        self.num_layers = 4
        self.encoder = TransformersBertEncoder(config, self.num_layers)
        self.embed_dim = 768
        self.top_k = 64
        # self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, batch_first = True)
        # self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers)
        # self.attentionlayer = torch.nn.MultiheadAttention(self.embed_dim, self.n_heads, batch_first = True)
        self.classification_head = torch.nn.Linear(self.embed_dim, 1)
    @ForwardContext.wrap
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        ###MODIFIED###
        embedding_output = input_ids
        embedding_output = self.invertible_adapters_forward(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        attention_result = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )[0]
        context_input = attention_result[:,0,:].unsqueeze(1)
        candidate_input = attention_result[:,1:,:]
        context_input = context_input.expand(-1, self.top_k, -1).reshape(-1, 1, context_input.size(-1))
        candidate_input = candidate_input.reshape(-1, context_input.size(-1), 1)
        scores = torch.bmm(context_input, candidate_input)
        scores = scores.reshape(input_ids.size(0), input_ids.size(1)-1)
        # elif self.params["classification_head"] == "linear":
        # scores = self.classification_head(attention_result)[:,1:,:]
        return scores.squeeze(-1)

class TransformersBertEncoder(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.num_layers = num_layers
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class IdentityInitializedTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, d_model, n_head, num_layers, dim_feedforward=3072, dropout=0.1, activation="relu"):
        super(IdentityInitializedTransformerEncoderLayer, self).__init__(d_model, n_head)
    def forward(self,src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        out1 = super(IdentityInitializedTransformerEncoderLayer, self).forward(src, src_mask, src_key_padding_mask, is_causal)
        out = out1 + src
        
        return out
class ExtendExtensionModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(ExtendExtensionModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )  
        self.params = params
        self.n_heads = params["n_heads"]
        self.embed_dim = 768
        self.num_layers = params["num_layers"]
        self.top_k = 64
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, dim_feedforward=self.embed_dim*4, batch_first = True)
        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers)
        self.attentionlayer = torch.nn.MultiheadAttention(self.embed_dim, self.n_heads, batch_first = True)
        self.classification_head = torch.nn.Linear(self.embed_dim, 1)
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)
        if params["identity_init"]:
            self.transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.n_heads, self.num_layers).to(self.device)
            self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers)

        # self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        # self.feedforward = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )
        print(self.transformerencoder)
    def forward(self, input):
        # attention_result = self.attentionlayer(input, input, input)[0]
        # attention_result = self.layer_norm(input+attention_result)
        # linear_result = self.feedforward(attention_result)
        # attention_result = self.layer_norm(linear_result+attention_result)
        token_type_context = torch.zeros(input.size(0), 1).int().to(self.device)
        token_embedding_context = self.token_type_embeddings(token_type_context)
        token_type_candidate = torch.ones(input.size(0), input.size(1)-1).int().to(self.device)
        token_embedding_candidate = self.token_type_embeddings(token_type_candidate)
        token_embedding_type = torch.cat([token_embedding_context, token_embedding_candidate], dim = 1)
        input += token_embedding_type
        attention_result = self.transformerencoder(input)
        if self.params["classification_head"] == "dot":
            context_input = attention_result[:,0,:].unsqueeze(1)
            candidate_input = attention_result[:,1:,:]
            context_input = context_input.expand(-1, self.top_k, -1).reshape(-1, 1, context_input.size(-1))
            candidate_input = candidate_input.reshape(-1, context_input.size(-1), 1)
            scores = torch.bmm(context_input, candidate_input)
            scores = scores.reshape(input.size(0), input.size(1)-1)
        elif self.params["classification_head"] == "linear":
            scores = self.classification_head(attention_result)[:,1:,:]

        return scores.squeeze(-1)
    def forward_chunk(self, xs, ys, dot = False, size_chunk = 1):
        xs = xs.repeat_interleave(size_chunk, dim=0)
        token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
        token_embedding_xs = self.token_type_embeddings(token_type_xs)
        token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
        token_embedding_ys = self.token_type_embeddings(token_type_ys)
        input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        attention_result = self.transformerencoder(input)
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,1:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        else:
            scores = self.linearhead(attention_result[:,1:,:])
            scores = scores.squeeze(-1)
        return scores
class ExtendExtensionRanker(torch.nn.Module):
    def __init__(self, params):
        super(ExtendExtensionRanker, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.params = params
        self.n_gpu = torch.cuda.device_count()
        self.tokenizer = None   
        if self.params["extend_bert"]:
            config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
            self.model = ExtendExtensionBertModule.from_pretrained("bert-base-cased", config = config)
            self.model.add_adapter('extend-adapter')
            self.model.active_adapters = 'extend-adapter'
            self.model = self.model.to(self.device)
        else:
            self.model = ExtendExtensionModule(params, self.tokenizer)
            self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, input, label_input, context_length, evaluate = False, hard_entire = False, beam_ratio =0.5, sampling = False):
        if hard_entire:
            round = 1
            top_k = self.params["sample_cands"]
            context = input[:,0,:].unsqueeze(1)
            candidate = input[:,1:,:]
            B, C, D = candidate.size()
            candidates_embeds = {}
            candidates_embeds["embeds"] = candidate
            candidates_embeds["idxs"] = torch.arange(candidates_embeds["embeds"].size(1))\
                        .expand(candidates_embeds["embeds"].size(0), -1)
            if self.params["classification_head"] == "dot":
                dot = True
            else: dot = False
            processed_tensor = candidate.reshape(-1, top_k, D)
            scores = self.model.forward_chunk(context, processed_tensor, dot, int(candidate.size(1)/top_k)) 
            scores = scores.view(B, -1, top_k)
            idxs = torch.topk(scores, int(top_k*beam_ratio))[1]
            cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
            idxs += cumulative_idxs
            idxs = idxs.view(B, int(C*beam_ratio))
            batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
            scores = scores.view(B, C)
            candidates_embeds["embeds"] = candidates_embeds["embeds"][batch_idxs, idxs, :]
            candidates_embeds["idxs"] = candidates_embeds["idxs"][batch_idxs, idxs.cpu()]
            while idxs.size(1) >= top_k:
                round += 1
                processed_tensor = candidates_embeds["embeds"].reshape(-1, top_k, candidates_embeds["embeds"].size(-1))
                idxs_chunks = candidates_embeds["idxs"].reshape(-1, top_k)
                new_scores = self.model.forward_chunk(context, processed_tensor, dot, int(candidates_embeds["embeds"].size(1)/top_k))
                new_scores = new_scores.view(B, -1, top_k)
                idxs = torch.topk(new_scores, int(top_k*beam_ratio))[1]
                cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
                idxs += cumulative_idxs
                idxs = idxs.view(B, int(candidates_embeds["embeds"].size(1)*beam_ratio))
                batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
                new_scores = new_scores.view(B, candidates_embeds["embeds"].size(1))
                if sampling:
                    for row in range(candidates_embeds["embeds"].size(0)):
                        scores[row, candidates_embeds["idxs"][row]] *= (round-1)/round
                        scores[row, candidates_embeds["idxs"][row]] += new_scores[row]/round
                else:
                    for row in range(candidates_embeds["embeds"].size(0)):
                        scores[row, candidates_embeds["idxs"][row]] += new_scores[row]

                candidates_embeds["embeds"] = candidates_embeds["embeds"][batch_idxs, idxs, :]
                candidates_embeds["idxs"] = candidates_embeds["idxs"][batch_idxs, idxs.cpu()]
            return scores
        else:
            scores = self.model(input)
            loss = self.criterion(scores, label_input)
        return loss, scores



'''DotProductModule and DotProductRanker'''
# input: same as mlpranker (batch_Size, top_k, 2, Embedding_Size) (e.g. (32, 64, 2, 768))
class AdditiveScoreModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(AdditiveScoreModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.input_size = 768
        self.params = params
        self.linear_layer_concat = torch.nn.Linear(self.input_size*2, self.input_size)
        self.linear_layer_mention = torch.nn.Linear(self.input_size, self.input_size)
        self.linear_layer_entity = torch.nn.Linear(self.input_size, self.input_size)
        self.linear_score = torch.nn.Linear(self.input_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        if self.params["operation"] == "concat":
            input = torch.cat([input[:,:,0,:], input[:,:,1,:]], dim = -1)
            scores = self.linear_score(self.tanh(self.linear_layer_concat(input)))
        elif self.params["operation"] == "addition":
            scores = self.linear_score(self.tanh(self.linear_layer_mention(input[:,:,0,:])+self.linear_layer_entity(input[:,:,1,:])))
        return scores.squeeze(-1)

class AdditiveScoreRanker(torch.nn.Module):
    def __init__(self, params):
        super(AdditiveScoreRanker, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        self.tokenizer = None   
        self.model = AdditiveScoreModule(params, self.tokenizer)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, input, label_input, context_length, evaluate = False):
        scores = self.model(input)
        loss = self.criterion(scores, label_input)
        return loss, scores

class MultiplicativeScoreModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(MultiplicativeScoreModule, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.input_size = 768
        self.params = params
        if params["initialization"] == "xavier":
            self.multiplicative_parameters  = nn.parameter.Parameter(torch.empty(1, self.input_size, self.input_size).to(self.device))
            nn.init.xavier_uniform_(self.multiplicative_parameters)
        elif params["initialization"] == "identity":
            self.multiplicative_parameters = torch.nn.Parameter(torch.eye(self.input_size).unsqueeze(0)) # (1, 768, 768)
        self.context_mapping = nn.Linear(self.input_size, self.input_size)
        self.entity_mapping = nn.Linear(self.input_size, self.input_size)
    def forward(self, input):
        if self.params["mapping"]:
            context = self.context_mapping(input[:,:,0,:].reshape(-1, input.size(-1)))
            entity = self.context_mapping(input[:,:,1,:].reshape(-1, input.size(-1)))
        else:
            context = input[:,:,0,:]
            entity = input[:,:,1,:]
        context = context.reshape(-1, 1, input.size(-1))
        entity = entity.reshape(-1, input.size(-1), 1)
        new_context = torch.bmm(context, self.multiplicative_parameters.expand(context.size(0), self.multiplicative_parameters.size(1), self.multiplicative_parameters.size(2)))
        scores = torch.bmm(new_context, entity).reshape(input.size(0), input.size(1))
        return scores

class MultiplicativeScoreRanker(torch.nn.Module):
    def __init__(self, params):
        super(MultiplicativeScoreRanker, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        self.tokenizer = None   
        self.model = MultiplicativeScoreModule(params, self.tokenizer)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, input, label_input, context_length, evaluate = False):
        scores = self.model(input)
        loss = self.criterion(scores, label_input)
        return loss, scores


class MlpwithBiEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(MlpwithBiEncoderModule, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        if params["architecture"] == "mlp_with_bert":
            self.mlpmodule = MlpModule(self.params).to(self.device)
        elif params["architecture"] == "mlp_with_som_finetuning":
            self.mlpwithsommodule = MlpwithSOMModule(self.params).to(self.device)
            # self.mlpwithsommodule = MlpwithSOMModule(self.params).to(self.device)
        
        if params["anncur"]:
            if params["adapter"]:
                config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
                ctxt_bert = BertModel.from_pretrained(params["bert_model"], config=config)
                cand_bert = BertModel.from_pretrained(params['bert_model'], config=config)
                ctxt_bert.add_adapter('ctxt-bert-adapter')
                cand_bert.add_adapter('cand-bert-adapter')
                ctxt_bert.train_adapter('ctxt-bert-adapter')
                cand_bert.train_adapter('cand-bert-adapter')
                ctxt_bert.set_active_adapters("ctxt-bert-adapter")
                cand_bert.set_active_adapters("cand-bert-adapter")
            else:
                config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
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
            tokenizer,
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            params = params
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            tokenizer,
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            params = params
        )
        if params["freeze_bert"]:
            for param in self.context_encoder.parameters():
                param.requires_grad = False
            for param in self.cand_encoder.parameters():
                param.requires_grad = False
            
        self.config = ctxt_bert.config
        self.params = params

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt  = None, mask_ctxt = None, token_idx_cands = None, segment_idx_cands = None, mask_cands = None, num_cand = None
    ):
        # Obtaining BERT embedding of context and candidate from bi-encoder
        # if token_idx_cands is None:
        #     print("*** For Evaluation Purpose ***")
        #     token_idx_cands = torch.randint(1, 3, (64, 128)).to(self.device)
        #     segment_idx_ctxt = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        #     segment_idx_cands = torch.zeros(token_idx_cands.shape).int().to(self.device)
        #     mask_ctxt = torch.zeros(token_idx_ctxt.shape).int().to(self.device)
        #     mask_cands = torch.zeros(token_idx_cands.shape).int().to(self.device)
        #     print(token_idx_ctxt.shape, segment_idx_ctxt.shape, mask_ctxt.shape)
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt, cls_ctxt, all_embeddings_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands, cls_cands, all_embeddings_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands, data_type="candidate"
            )
            if self.params["architecture"] == "mlp_with_som_finetuning":
                if self.params["bert_model"] == "bert-base-cased":
                    bert_embedding_size = 768
                elif self.params["bert_model"] == "bert-large-uncased":
                    bert_embedding_size = 1024
                all_embeddings_ctxt = all_embeddings_ctxt.reshape(-1, self.params["max_context_length"], bert_embedding_size)
                all_embeddings_cands = all_embeddings_cands.reshape(-1, self.params["max_cand_length"], bert_embedding_size)
                input = torch.stack((all_embeddings_ctxt, all_embeddings_cands), dim = 1)
                score = self.mlpwithsommodule(input)
            elif self.params["architecture"] == "mlp_with_bert":
                score = self.mlpmodule(torch.cat((embedding_ctxt.unsqueeze(dim = 1), embedding_cands.unsqueeze(dim = 1)), dim=1).unsqueeze(dim = 0))
        return score

class MlpwithBiEncoderRanker(BiEncoderRanker): 
    def __init__(self, params, shared=None):
        super(MlpwithBiEncoderRanker, self).__init__(params)
        if params["path_to_mlpmodel"] is not None:
            if params["architecture"] == "mlp_with_bert":
                # print("load mlp")
                # print("previous,",self.model.state_dict().items())
                model_path = params.get("path_to_mlpmodel", None)
                state_dict = torch.load(model_path)['model_state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('model.','module.mlpmodule.' )
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict, strict = False)
                # print("after,", self.model.state_dict().items())
            elif params["architecture"] == "mlp_with_som_finetuning":
                model_path = params.get("path_to_mlpmodel", None)
                state_dict = torch.load(model_path)['model_state_dict']

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('model.module.','module.mlpwithsommodule.' )
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict, strict = False)
            
            # self.print_parameters()
    def print_parameters(self):
        for name, param in self.model.named_parameters():
            print (name, param.data)
    def build_model(self):
        self.model = MlpwithBiEncoderModule(self.params, self.tokenizer)
    def load_model(self, fname, cpu=False):
        if self.params["anncur"]:
            if cpu:
                state_dict = torch.load(fname, map_location= "cpu")["state_dict"]
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
            if cpu:
                state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
            else:
                state_dict = torch.load(fname)
            self.model.load_state_dict(state_dict)

    def forward(self, input_idx, label_input, context_len, evaluate = False):
        scores = self.score_candidate(input_idx, context_len)
        # print(input_idx.shape)
        # print(scores.shape)
        # print(label_input.shape)
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


    def score_candidate(self, text_vecs, context_len):
        # Pre-processing input for MlpwithBiEncoderModule
        num_cand = text_vecs.size(1) # 64
        text_vecs_ctxt = text_vecs[:,:,0,:].squeeze(dim = 2)
        batch_size = text_vecs_ctxt.size(0)
        text_vecs_ctxt = text_vecs_ctxt.view(-1, text_vecs_ctxt.size(-1))
        text_vecs_cands = text_vecs[:,:,1,:].squeeze(dim = 2)
        text_vecs_cands = text_vecs_cands.view(-1, text_vecs_cands.size(-1))
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            text_vecs_ctxt.int(), self.NULL_IDX, context_len,
        )
        token_idx_cands, segment_idx_cands, mask_cands = self.to_bert_input(
            text_vecs_cands.int(), self.NULL_IDX, context_len,
        )
        # Get BERT embeddings
        score = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, token_idx_cands, segment_idx_cands, mask_cands, num_cand 
        )
        # Take them as input of MLP layers
        return score.view(-1, num_cand)

class MlpwithSOMModule(nn.Module):
    def __init__(self, params):
        super(MlpwithSOMModule, self).__init__()
        self.params = params
        self.input_size = 768*2
        self.mlpmodule = MlpModule(self.params)
    def forward(self, input):
        if input.ndim == 5:
            entity = input[:,:,1,:,:]
            context = input[:,:,0,:,:]
            entity = entity.reshape(-1, entity.size(-2), entity.size(-1))
            context = context.reshape(-1, entity.size(-2), entity.size(-1))
        else:
            entity = context[:,1,:,:]
            context = context[:,0,:,:] #(batch*top_k,max_length, embedding_dimension)
        mask = torch.tensor(context != 0)[:,:,0]
        if self.params["ffnn_over_all"]:
            context = context.unsqueeze(-2)
            entity = entity.unsqueeze(-3)

            context = context.expand(-1, -1, context.size(1), -1).cpu()
            entity = entity.expand(-1, entity.size(2), -1, -1).cpu()

            context = context.reshape(context.size(0)*context.size(1)*context.size(2), context.size(3))
            entity = entity.reshape(entity.size(0)*entity.size(1)*entity.size(2), entity.size(3))
            mlp_input = torch.stack((context, entity), dim = 1)
            output = self.mlpmodule(mlp_input)
            
            output = output.reshape(input.size(0)*input.size(1), input.size(3), input.size(3))
            output = torch.max(output, dim = -1)[0]
            output *= mask
            # print(output, mask)
            output = torch.sum(output, -1)
            # print("0", output.shape)
            return output, None
        output = torch.bmm(context, entity.transpose(1,2))
        if self.params["colbert_baseline"]:
            scores = torch.max(output, dim = -1)[0]
            scores = scores.reshape(input.size(0), input.size(1), input.size(3))
            scores = torch.sum(scores, dim = -1)
            return scores, output
        else:
            # reshape the output tensor to have shape (128, 128)
            dot_scores, argmax_values = torch.max(output, dim=-1)
            # print(entity[argmax_values].shape)
            mlp_input = torch.stack([context, torch.gather(entity, dim = 1, index = argmax_values.unsqueeze(-1).expand(-1,-1,entity.size(-1)))], dim = -2)
            scores = self.mlpmodule(mlp_input).squeeze(-1)
        scores *= mask
        dot_scores *= mask
        scores = torch.sum(scores, -1)
        dot_scores = torch.sum(dot_scores, -1).reshape(input.size(0), input.size(1))
        return scores, dot_scores


class MlpwithSOMModuleCosSimilarity(nn.Module):
    def __init__(self, input_size):
        super(MlpwithSOMModuleCosSimilarity, self).__init__()
        self.cossimilarity = nn.CosineSimilarity(dim = -1)
        self.input_size = input_size
        self.mlpmodule = MlpModule(self.input_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    def forward(self, context):
        eps = 1e-8
        entity = context[:,1,:,:]
        context = context[:,0,:,:] #(batch*top_k,max_length, embedding_dimension)
       
        batch_size = entity.size(0)
        max_length = entity.size(1)
        embedding_dimension = entity.size(2)

        entity = torch.functional.normalize
        # perform batch-wise dot product using torch.bmm
        # output = self.cossimilarity(context.unsqueeze(2), entity.unsqueeze(1))
        # print(context, entity)
        # context_norm = torch.tensor(np.linalg.norm(context.detach().cpu().numpy(), axis = -1, keepdims = True)).to(self.device)
        # entity_norm = torch.tensor(np.linalg.norm(entity.detach().cpu().numpy(), axis = -1, keepdims = True)).to(self.device)

        # context_norm = torch.linalg.norm(context, dim = -1, keepdim = True).clone()
        # entity_norm = torch.linalg.norm(entity, dim = -1, keepdim = True).clone()
        # print("1", context_norm, entity_norm)
        
        # denominator = context_norm@entity_norm.transpose(1,2)+eps
        # context/=context_norm
        # entity/=entity_norm
        # print("denominator", denominator, denominator.shape)
        output = torch.bmm(context, entity.transpose(1,2))
        
        # print("output shape 1", output)
        # reshape the output tensor to have shape (128, 128)
        argmax_values = torch.argmax(output, dim=-1)
        # print(entity[argmax_values].shape)
        input = torch.stack([context, torch.gather(entity, dim =1, index = argmax_values.unsqueeze(-1).expand(-1,-1,embedding_dimension))], dim = -2)
        output = torch.sum(self.mlpmodule(input), -2)
        # print("output shape", output.shape)
        return output.squeeze(-1)

class MlpwithSOMRanker(CrossEncoderRanker): 
    def __init__(self, params, shared=None):
        super(MlpwithSOMRanker, self).__init__(params)
        self.loss_weight = torch.tensor([self.params["loss_weight"]]).to(self.device)
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.KLDivLoss(reduction = "batchmean")
        self.softmax = nn.Softmax(dim = 1)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
    def build_model(self):
        if self.params["cos_similarity"]:
            self.model = MlpwithSOMModuleCosSimilarity(self.params)
        else:
            self.model = MlpwithSOMModule(self.params)
    def forward(self, input, label_input, context_length, evaluate = False):
        if self.params["dot_product"]:
            num_cand = input.size(1)
            scores, dot_product = self.model(input)
            scores = scores.view(-1, num_cand)
            loss1 = self.criterion1(scores, label_input)
            loss2 = self.criterion2(self.logsoftmax(scores), self.softmax(dot_product))
            loss = loss1 + self.loss_weight*loss2
            return loss, scores, loss1, loss2
        else:
            num_cand = input.size(1)
            scores, _ = self.model(input)
            scores = scores.view(-1, num_cand)
            loss = F.cross_entropy(scores, label_input)
        return loss, scores
    
class SOMModule(nn.Module):
    def __init__(self):
        super(SOMModule, self).__init__()
    def forward(self, context):
        batch_size = context.size(0)
        top_k = context.size(1)
        max_length = context.size(3)
        embedding_dimension = context.size(4)

        entity = context[:,:,1,:,:]
        context = context[:,:,0,:,:] #(max_length, embedding_dimension)

        entity = entity.reshape(-1, max_length, embedding_dimension)
        context = context.reshape(-1, max_length, embedding_dimension)
        # perform batch-wise dot product using torch.bmm
        output = torch.bmm(context, entity.transpose(1,2)).squeeze().reshape(batch_size, top_k, max_length, max_length)
        # reshape the output tensor to have shape (128, 128)
        output = output.reshape(batch_size, top_k, max_length, max_length)
        context = context.reshape(batch_size, top_k, max_length, embedding_dimension)

        entity = entity.reshape(batch_size, top_k, max_length, embedding_dimension)
        argmax_values = torch.argmax(output, dim=-1)
        # print(entity[argmax_values].shape)
        entity = torch.gather(entity, dim =2, index = argmax_values.unsqueeze(-1).expand(-1,-1,-1,embedding_dimension))
        entity = entity.reshape(-1, embedding_dimension)
        context = context.reshape(-1, embedding_dimension)
        
        output = torch.bmm(context.unsqueeze(dim = -2), entity.unsqueeze(dim = -1))
        output = output.reshape(batch_size, top_k, max_length)
        output = torch.sum(output, -1)

        return output

class SOMRanker(CrossEncoderRanker): 
    def __init__(self, params, shared=None):
        super(SOMRanker, self).__init__(params)
    def build_model(self):
        self.model = SOMModule()
    def forward(self, input, label_input, context_length, evaluate = False):
        scores = self.model(input)
        loss = F.cross_entropy(scores, label_input)
        return loss, scores
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


# class MlpwithSOM()