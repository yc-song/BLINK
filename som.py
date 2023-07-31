import torch
import torch.nn as nn
baseline = False
ffnn_over_all = True
multi_task = False
class MlpModule(nn.Module):
    def __init__(self, model_input):
        super(MlpModule, self).__init__()
        self.input_size=model_input 
        self.input_size = model_input
        self.fc=nn.Linear(self.input_size, self.input_size)
        self.fc2=nn.Linear(self.input_size, 1)
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        self.act_fn = nn.Tanh()
        self.current_dim = self.input_size
        self.dim_red = model_input
        self.layers.append(nn.Linear(int(self.current_dim), int(self.dim_red)))
        self.current_dim = model_input
        for i in range(5):
            self.layers.append(nn.Linear(int(self.current_dim), int(self.current_dim)))

        self.layers.append(nn.Linear(int(self.current_dim), 1))

    def forward(self, input):
        print(self.layers)
        input = torch.flatten(input, start_dim = -2)
        for i, layer in enumerate(self.layers[:-1]):
            input = self.act_fn(layer(self.dropout(input)))
        input = self.layers[-1](self.dropout(input))
        return input

class MlpwithSOMModule(nn.Module):
    def __init__(self, input_size):
        super(MlpwithSOMModule, self).__init__()
        self.input_size = input_size
        self.mlpmodule = MlpModule(self.input_size)
    def forward(self, input):
        self.mlpmodule.eval()
        entity = input[:,:,1,:,:]
        context = input[:,:,0,:,:] #(max_length, embedding_dimension)
        entity = entity.reshape(-1, entity.size(-2), entity.size(-1))
        context = context.reshape(-1, entity.size(-2), entity.size(-1))
        mask = torch.tensor(context != 0)[:,:,0]
        if ffnn_over_all:
            context = context.unsqueeze(-2)
            entity = entity.unsqueeze(-3)

            context = context.expand(-1, -1, context.size(1), -1)
            entity = entity.expand(-1, entity.size(2), -1, -1)

            context = context.reshape(context.size(0)*context.size(1)*context.size(2), context.size(3))
            entity = entity.reshape(entity.size(0)*entity.size(1)*entity.size(2), entity.size(3))
            mlp_input = torch.stack((context, entity), dim = 1)
            output = self.mlpmodule(mlp_input)
            
            output = output.reshape(input.size(0)*input.size(1), input.size(3), input.size(3))
            print("mlp score", output, output.shape)
            # count = 0
            # for i in range(input.size(0)):
            #     for j in range(input.size(1)):
            #         for k in range(input.size(3)):
            #             for u in range(input.size(3)):
            #                 mlp_input2 = torch.stack((input[i,j,0,k,:], input[i,j,1,u,:]), dim = 0)
            #                 # print(mlp_input[count] == mlp_input2)
            #                 mlp_output = self.mlpmodule(mlp_input2)
            #                 impl_output = output[i*input.size(1)+j][k][u]
            #                 count+=1
            #                 print(mlp_output)
            #                 # print(impl_output)
            #                 print(mlp_output.item() == impl_output)
            output = torch.max(output, dim = -1)[0]
            output *= mask
            # print(output, mask)
            output = torch.sum(output, -1)
            # print("0", output.shape)

            return output, None
        # reshape the output tensor to have shape (128, 128)
        elif baseline:
            output = torch.bmm(context, entity.transpose(1,2)) # (6, 4, 4)
            output = torch.max(output, dim = -1)[0]
            output = output.reshape(input.size(0), input.size(1), input.size(3))
            print("score", output.shape) 
            print(output.shape)
            output = torch.sum(output, dim = -1)
            print("o", output)

            return output, None
        else: 
            output = torch.bmm(context, entity.transpose(1,2)) # (6, 4, 4)
            print("output", output.shape)
            dot_scores, argmax_values = torch.max(output, dim=-1)
            print("argmax", dot_scores.shape)
            # print(entity[argmax_values].shape)
            mlp_input = torch.stack([context, torch.gather(entity, dim = 1, index = argmax_values.unsqueeze(-1).expand(-1,-1,entity.size(-1)))], dim = -2)
            print("input", input)
            scores = self.mlpmodule(mlp_input).squeeze(-1) # (6, 4)
            print("scores previous", scores.shape)
        scores *= mask
        dot_scores *= mask
        print("scores", scores)
        scores = torch.sum(scores, -1)
        dot_scores = torch.sum(dot_scores, -1).reshape(input.size(0), input.size(1))
        print("output", dot_scores.shape)
        return scores, dot_scores


class MlpwithSOMCosSimilarityModule(nn.Module):
    def __init__(self, input_size):
        super(MlpwithSOMCosSimilarityModule, self).__init__()
        self.cossimilarity = nn.CosineSimilarity(dim = -1)
        self.input_size = input_size
        self.mlpmodule = MlpModule(self.input_size)
        
    def forward(self, context):
        eps = 1e-8
        entity = context[:,:,1,:,:]
        context = context[:,:,0,:,:] #(max_length, embedding_dimension)
        entity = entity.reshape(-1, max_length, embedding_dimension)
        context = context.reshape(-1, max_length, embedding_dimension)

        # perform batch-wise dot product using torch.bmm
        # output = self.cossimilarity(context.unsqueeze(2), entity.unsqueeze(1))
        print(context, entity)
        context_norm = torch.linalg.norm(context, dim = -1, keepdim = True)
        entity_norm = torch.linalg.norm(entity, dim = -1, keepdim = True)
        print("1", context_norm, entity_norm)

        denominator = context_norm@entity_norm.transpose(1,2)
        print("denominator", denominator, denominator.shape)
        print("2", torch.bmm(context, entity.transpose(1,2)).shape)
        output = torch.bmm(context, entity.transpose(1,2))/denominator
        context/=context_norm
        entity/=entity_norm
        print("output shape 1", output)
        # reshape the output tensor to have shape (128, 128)
        output = output.reshape(batch_size, top_k, max_length, max_length)
        print("output shape 2", output)
        context = context.reshape(batch_size, top_k, max_length, embedding_dimension)

        entity = entity.reshape(batch_size, top_k, max_length, embedding_dimension)
        argmax_values = torch.argmax(output, dim=-1)
        # print(entity[argmax_values].shape)
        input = torch.stack([context, torch.gather(entity, dim =2, index = argmax_values.unsqueeze(-1).expand(-1,-1,-1,embedding_dimension))], dim = -2)
        print("input shape", input.shape)
        output = torch.sum(self.mlpmodule(input), -2)
        print("output shape", output.shape)

        return output.squeeze(-1)
class MlpwithSOMRanker(torch.nn.Module): 
    def __init__(self, shared=None):
        super(MlpwithSOMRanker, self).__init__()
        self.build_model()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.alpha = 1
    def build_model(self):
        self.model = MlpwithSOMModule(10)
    def forward(self, input, label_input, context_length, evaluate = False):
        print("input shape", input.shape)
        num_cand = input.size(1)
        loss_weight = torch.tensor([1])
        outputs, scores = self.model(input) #outputs: MLP output, scores:dot product output
        outputs = outputs.view(-1, num_cand)
        criterion1 = torch.nn.CrossEntropyLoss()
        loss1=criterion1(outputs, label_input)
        print(outputs, scores, "11")
        if multi_task:
            criterion2 = nn.KLDivLoss(reduction = "batchmean")
            loss2 = criterion2(self.logsoftmax(outputs), self.softmax(scores))
            loss = loss1+self.alpha*loss2
            return loss, outputs, loss1, loss2
        else: 
            loss = loss1
        return loss, scores, None, None
        # else:
        #     num_cand = input.size(1)
        #     scores, _ = self.model(input)
        #     scores = scores.view(-1, num_cand)
        #     loss = F.cross_entropy(scores, label_input)
        # return loss, scores
device = torch.device('cuda')

batch_size = 2
top_k = 3
max_length = 4
embedding_dimension = 5
### modify ###
context = torch.tensor([[[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], \
[[1., 0., 0., 0., 1.], [1., 1., 0., 1., 1.], [1., 1., 1., 1., 0.], [1., 1., 1., 1., 1.]]])

entity = torch.tensor([[[[1., 0., 0., 1., 1.], [1., 0., 1., 0., 0.], [1., 0., 0., 0., 0.], [1., 1., 1., 1., 0.]], \
[[1., 0., 0., 1., 0.], [1., 0., 1., 1., 0.], [1., 0., 1., 0., 1.], [0., 0., 0., 0., 0.]],\
[[1., 0., 0., 1., 0.], [1., 1., 1., 0., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]],\
[[[1., 0., 0., 0., 1.], [1., 1., 0., 0., 0.], [1., 0., 0., 0., 1.], [1., 1., 0., 1., 1.]], \
[[1., 1., 1., 1., 0.], [1., 0., 1., 1., 1.], [1., 0., 0., 1., 1.], [1., 1., 0., 0., 0.]], \
[[1., 0., 0., 1., 1.], [1., 0., 1., 0., 0.], [1., 1., 1., 0., 1.], [0., 0., 0., 0., 0.]]]])

print(context.shape, entity.shape) 
# context = torch.randn((batch_size, max_length, embedding_dimension))
# entity = torch.randn((batch_size, top_k, max_length, embedding_dimension))
# print(entity)
context = context.unsqueeze(1).expand(-1, top_k, -1, -1)
context = torch.stack((context, entity), dim = 2)   
print(context.shape)
mlpmodule = MlpwithSOMModule(10)
mlpranker = MlpwithSOMRanker()
label_input = torch.randint(1, 3, (batch_size,))
print(mlpranker(context, label_input, 128))
