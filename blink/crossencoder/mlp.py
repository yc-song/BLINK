import torch
from torch import nn, optim
import torch.nn.functional as F
import os
# from pytorch_transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
import sys
sys.path.append('/mnt/f/BLINK')
from blink.common.ranker_base import get_model_obj
from transformers.models.bert.tokenization_bert import BertTokenizer
# from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from blink.common.params import BlinkParser
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.common.ranker_base import BertEncoder, get_model_obj

def load_mlp(params):
    # Init model
    crossencoder = MlpModel(params)
    return crossencoder

input_size=768*2 #(64,256)
## model이랑 module, ranker 나눠서 modeule이랑 ranker에서 model 호출
class MlpModel(nn.Module):
    def __init__(self,params, model_input=input):
        super(MlpModel, self).__init__()
        self.params=params
        self.device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        self.n_gpu = torch.cuda.device_count()
        self.data_parallel=1
        self.tokenizer=BertTokenizer.from_pretrained(
                "bert-base-cased"
            )
        self.build_model()
        self.top_k = params["top_k"]
        if params["path_to_model"] is not None:
            print("load")
            self.load_model(params["path_to_model"])
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
    def build_model(self):
        self.model=MlpModule(self.params)
        if self.params["architecture"] == "mlp_with_bert":
            self.model = MlpwithBERTModule(self.params, self.tokenizer)
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model.state_dict(), output_dir)
    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")["model_state_dict"]
        else:
            state_dict = torch.load(fname)["model_state_dict"]
        self.model.load_state_dict(state_dict)
    def forward(self, input, label_input, context_length,  bi_encoder_score = None, evaluate = False):
        # summary(self.model, input_size=(1, 2, 1024))

        # input shape: (batch size, top_k, 2, bert_hidden_dimension)
        # score shape: (batch_size, top_k)
        loss1= None
        loss2 = None
        if not self.params["sampling"]:
            if self.params["dot_product"]:
                loss_weight = torch.tensor([self.params["loss_weight"]]).to(self.device)
                context = input[:,:,0,:].reshape(-1, 1, input.size(-1))
                entity = input[:,:,1,:].reshape(-1, input.size(-1), 1)
                softmax = nn.Softmax(dim=1)
                logsoftmax = nn.LogSoftmax(dim = 1)
                dot_product = torch.bmm(context, entity).reshape(input.size(0), input.size(1))
                scores = self.model(input).squeeze(dim = -1)
                criterion1 = torch.nn.CrossEntropyLoss()
                loss1=criterion1(scores, label_input)
                criterion2 = nn.KLDivLoss(reduction = "batchmean")
                loss2 = criterion2(logsoftmax(scores), softmax(dot_product))
                loss = loss1 + loss_weight*loss2
                return loss, scores, loss1, loss2
            else:
                if self.params["architecture"] == "mlp_with_bert":

                    scores=torch.squeeze(self.model(input[:,:,0,:], input[:,:,1,:]), dim=2)
                else:
                    scores=torch.squeeze(self.model(input), dim=2)
                if self.params["binary_loss"]:
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                
                loss=criterion(scores, label_input)
        else:    
            if not evaluate:
                num_samples = self.params["num_samples"] # the number of negative samples
                label_input = torch.unsqueeze(label_input, dim = 1)
                # Making target tensor for BCELoss
                # Target tensor shape: (batch_size, num_samples + 1)
                # Target tensor looks like: [[1, 0, ... , 0], ... [1, 0, ..., 0]]
                target = torch.zeros((input.size(0), num_samples + 1), device = torch.device('cuda'), dtype = torch.float32) 
                target[torch.arange(target.size(0)),0] = 1 
                # sampled input shape: (batch_size, num_samples + 1, 2, hidden_dim)
                # sampled input consists of gold_input and negative_input
                # gold_input : label-th tensor from each candidate
                gold_input = torch.gather(input, 1, label_input.view(input.size(0), 1, 1, 1).expand(input.size(0), 1, input.size(2), input.size(3)))
                masked_input = torch.ones_like(input).scatter(1, label_input.view(input.size(0), 1, 1, 1).expand(input.size(0), 1, input.size(2), input.size(3)), 0) # negative sample에 1 assign
                idxs_ = masked_input.nonzero()[:, 1].reshape(-1, input.size(1) - label_input.size(1), input.size(2), input.size(3))
                torch.set_printoptions(threshold = 1000000)
                negative_input = torch.gather(input, 1, idxs_)
                # negative input: negative sample tensor whose shape is (batch_size, num_samples, 2, hidden_dim)
                if self.params["hard_negative"] and bi_encoder_score is not None:
                    softmax = torch.nn.Softmax(dim = 1)
                    masked_score = torch.ones_like(bi_encoder_score).scatter(1, label_input.view(bi_encoder_score.size(0), 1).expand(bi_encoder_score.size(0), 1), 0) # negative sample에 1 assign
                    idxs_ = masked_score.nonzero()[:, 1].reshape(-1, masked_score.size(1) - label_input.size(1))
                    bi_encoder_score = torch.gather(bi_encoder_score, 1, idxs_)
                    bi_encoder_score = softmax(bi_encoder_score)
                    samples = torch.multinomial(bi_encoder_score, num_samples)
                    samples = samples.unsqueeze(2).unsqueeze(3).expand(-1 , -1 , 2 , 768)
                    negative_input = torch.gather(negative_input, 1, samples)
                    sampled_input = torch.cat((gold_input, negative_input), dim = 1)
                else:
                    # uniform sampling using 'torch.randperm'
                    negative_indices = torch.randperm(self.top_k-1)[:num_samples]
                    negative_input = negative_input[:,negative_indices,:,:] 
                    # concatenate gold_input and negative_input
                    sampled_input = torch.cat((gold_input, negative_input), dim = 1)
                    # Get scores by feeding sampled_input
                scores = torch.squeeze(self.model(sampled_input), dim = 2)
                # Assigning weights on BCELoss
                # 0.1 for negatives and 1 for gold
                if self.params["binary_loss"]:
                    weights = (0.1)*torch.ones((input.size(0), num_samples + 1), device = torch.device('cuda'))
                    weights[torch.arange(weights.size(0)),0] = 1
                    criterion = torch.nn.BCEWithLogitsLoss(weight = weights)
                else:
                    weights = (0.1)*torch.ones((num_samples + 1), device = torch.device('cuda'))
                    weights[0] = 1
                    criterion = torch.nn.CrossEntropyLoss(weight = weights)
                loss = criterion(scores, target)

            else:
                scores=torch.squeeze(self.model(input), dim=2)
                loss=F.cross_entropy(scores, label_input, reduction="mean")
        torch.set_printoptions(threshold=10_000)
        return loss, scores
class MlpModule(nn.Module):
    def __init__(self,params, model_input=input_size):
        super(MlpModule, self).__init__()
        self.params=params
        self.input_size=model_input 
        if params["bert_model"]=="bert-base-cased":
            self.input_size = 768*2
        self.fc=nn.Linear(self.input_size, self.input_size)
        self.fc2=nn.Linear(self.input_size, 1)
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        self.device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.positional_encoding = torch.nn.Parameter(torch.normal(0, 0.1, size=(2,768)).to(self.device))
        self.positional_encoding.requires_grad = True
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

    def forward(self, input):
        if self.params["positional_encoding"]:
            input += self.positional_encoding
        input = torch.flatten(input, start_dim = -2)
        for i, layer in enumerate(self.layers[:-1]):
            input = self.act_fn(layer(self.dropout(input)))
        input = self.layers[-1](self.dropout(input))
        return input

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        x_batch = x.reshape(x.size(0)*x.size(1)*x.size(2), x.size(3))
        batch_size, seq_len = x_batch.size()
        # [batch_size = 128, seq_len = 30]
        return self.encoding[:seq_len, :]

        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         
