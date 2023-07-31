import torch
import torch.nn as nn
class MlpModule(nn.Module):
    def __init__(self, model_input):
        super(MlpModule, self).__init__()
        self.input_size=model_input 
        self.input_size = 768*2
        self.fc=nn.Linear(self.input_size, self.input_size)
        self.fc2=nn.Linear(self.input_size, 1)
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        self.act_fn = nn.Tanh()
        self.current_dim = self.input_size
        self.dim_red = 768
        self.layers.append(nn.Linear(int(self.current_dim), int(self.dim_red)))
        self.current_dim = 768
        for i in range(5):
            self.layers.append(nn.Linear(int(self.current_dim), int(self.current_dim/2)))
            self.current_dim /= 2

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
    def forward(self, context):
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
        input = torch.stack([context, torch.gather(entity, dim =2, index = argmax_values.unsqueeze(-1).expand(-1,-1,-1,embedding_dimension))], dim = -2)

        output = torch.sum(self.mlpmodule(input), -2)
        return output.squeeze(-1)
device = torch.device('cuda')

batch_size = 16
top_k = 64
max_length = 128
embedding_dimension = 768
### modify ###
context = torch.rand(batch_size, max_length, embedding_dimension)
entity = torch.rand(batch_size, top_k, max_length, embedding_dimension)
context = context.unsqueeze(1).expand(-1, top_k, -1, -1)
context = torch.stack((context, entity), dim = 2)   
mlpmodule = MlpwithSOMModule(1536)
print(mlpmodule(context).shape)