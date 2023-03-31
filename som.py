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
        print("context", context, context.shape)
        print("entity", entity, entity.shape)
        entity = torch.gather(entity, dim =2, index = argmax_values.unsqueeze(-1).expand(-1,-1,-1,embedding_dimension))
        print("max entity", entity)
        entity = entity.reshape(-1, embedding_dimension)
        context = context.reshape(-1, embedding_dimension)
        
        output = torch.bmm(context.unsqueeze(dim = -2), entity.unsqueeze(dim = -1))
        output = output.reshape(batch_size, top_k, max_length)
        print("output", output, output.shape)
        output = torch.sum(output, -1)
        print("output", output, output.shape)

        return output
device = torch.device('cuda')

batch_size = 2
top_k = 3
max_length = 4
embedding_dimension = 5
### modify ###
context = torch.randint(2,(batch_size, max_length, embedding_dimension))
entity = torch.randint(2, (batch_size, top_k, max_length, embedding_dimension))
context = context.unsqueeze(1).expand(-1, top_k, -1, -1)
context = torch.stack((context, entity), dim = 2)   
mlpmodule = MlpwithSOMModule(1536)
print(mlpmodule(context).shape)
