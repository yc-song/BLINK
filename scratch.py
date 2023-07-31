import torch
# import gc
# gc.collect()
# torch.cuda.empty_cache()
# fname = "/home/jongsong/BLINK/models/zeshel/top200_candidates/train_special_tokens.t7"
# train_data = torch.load(fname)
# context_input = train_data["context_vecs"][:10]
# candidate_input = train_data["candidate_vecs"]
# world = train_data["worlds"][:10]
# idxs = train_data["indexes"][:10]
# label_input = train_data["labels"][:10]
# bi_encoder_score = train_data["nn_scores"][:10]
# context_input=context_input.unsqueeze(dim=1).expand(-1,200,-1)
# cand_list = []
# for i in range(context_input.size(0)):
#     cand_list.append(candidate_input[world[i].item()][idxs[i]].squeeze(dim = 0).tolist())
# cand_list = torch.FloatTensor(cand_list)
# print(cand_list.shape)
# context_input = torch.rand((16, 768))
# candidate_index = torch.randint(1, 4,(16, 64))
# candidate_input = torch.rand((5, 768))
# c = candidate_input[candidate_index]
# result = torch.cat([context_input.unsqueeze(-2), c], dim = -2)
# selfattention = torch.nn.MultiheadAttention(768, 8, batch_first = True)
# layer_norm = torch.nn.LayerNorm(768)
# linear_layer = torch.nn.Linear(768, 1)
# attention_embedding = selfattention(result, result, result)[0]
# attention_embedding = layer_norm(result + attention_embedding)
# print(attention_embedding.shape)
# result = linear_layer(attention_embedding)[:,1:,:]
# print(result.shape)

a = torch.randint(0, 2, (2005, 4, 5))
b = torch.randint(0, 2, (109, 4, 5))
scores_upper = None
# print(a)
# print(b)
batch_size = 64
for i in range(0, b.size(0), batch_size):
    batch = b[i:i+batch_size]
    print(batch.shape)
    a_reshaped = a.reshape((a.size(0)*a.size(1), a.size(2)))
    b_reshaped = batch.reshape((batch.size(0)*batch.size(1), batch.size(2)))
    c = torch.mm(a_reshaped, b_reshaped.t())
    c_reshaped = c.reshape(a.size(0)*a.size(1), batch.size(0), batch.size(1)).max(-1)[0]

    c_reshaped = c_reshaped.t().reshape(a.size(0)*batch.size(0), a.size(1))
    if scores_upper is None:
        scores_upper = torch.sum(c_reshaped, dim = -1).reshape(batch.size(0), a.size(0)).t()
    else:
        scores_upper = torch.cat([scores_upper, torch.sum(c_reshaped, dim = -1).reshape(batch.size(0), a.size(0)).t()], dim = -1)
    print(scores_upper)
    print(scores_upper.shape)

batch_size_a = a.size(0)
batch_size_b = b.size(0)   
# (64, 1, 128, 768)
a_reshaped_new = a.unsqueeze(1)
# (1, 64, 128, 768)
b_reshaped_new = b.unsqueeze(0)
# expand to (64, 64, 128, 768)
a_reshaped_new = a_reshaped_new.expand(batch_size_a ,batch_size_b, a.size(1), a.size(2))
b_reshaped_new = b_reshaped_new.expand(batch_size_a, batch_size_b, a.size(1), a.size(2))
# reshape to (64*64, 128, 768)
a_reshaped_new = a_reshaped_new.reshape(batch_size_a*batch_size_b, a.size(1), a.size(2))
b_reshaped_new = b_reshaped_new.reshape(batch_size_a*batch_size_b, a.size(1), a.size(2))
output = torch.bmm(a_reshaped_new, b_reshaped_new.transpose(1, 2))
scores = torch.max(output, dim = -1)[0]
scores = torch.sum(scores, dim = -1)
scores = scores.reshape(batch_size_a, batch_size_b)
print(scores, scores.shape)
print(scores_upper == scores)