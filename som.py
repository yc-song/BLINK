import torch
import torch.nn as nn
class MlpModule(nn.Module):
    def __init__(self, model_input):
        super(MlpModule, self).__init__()
        self.input_size=model_input 
        self.input_size = 5*2
        self.fc=nn.Linear(self.input_size, self.input_size)
        self.fc2=nn.Linear(self.input_size, 1)
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        self.act_fn = nn.Tanh()
        self.current_dim = self.input_size
        self.dim_red = 5
        self.layers.append(nn.Linear(int(self.current_dim), int(self.dim_red)))
        self.current_dim = 5
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
        batch_size = entity.size(0)
        top_k = entity.size(1)
        max_length = entity.size(2)
        embedding_dimension = entity.size(3)
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

        return output

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

        return output
device = torch.device('cuda')

batch_size = 128
top_k = 64
max_length = 128
embedding_dimension = 768
### modify ###
context = torch.randint(2,(batch_size, max_length, embedding_dimension))*1.0
entity = torch.randint(2, (batch_size, top_k, max_length, embedding_dimension))*1.0
# context = torch.randn((batch_size, max_length, embedding_dimension))
# entity = torch.randn((batch_size, top_k, max_length, embedding_dimension))

context = context.unsqueeze(1).expand(-1, top_k, -1, -1)
context = torch.stack((context, entity), dim = 2)   
print(context.shape)
raise("error")
mlpmodule = MlpwithSOMCosSimilarityModule(10)
print(mlpmodule(context).shape)
