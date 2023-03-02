import torch
import gc
gc.collect()
torch.cuda.empty_cache()
fname = "/home/jongsong/BLINK/models/zeshel/top200_candidates/train_special_tokens.t7"
train_data = torch.load(fname)
context_input = train_data["context_vecs"][:10]
candidate_input = train_data["candidate_vecs"]
world = train_data["worlds"][:10]
idxs = train_data["indexes"][:10]
label_input = train_data["labels"][:10]
bi_encoder_score = train_data["nn_scores"][:10]
context_input=context_input.unsqueeze(dim=1).expand(-1,200,-1)
cand_list = []
for i in range(context_input.size(0)):
    cand_list.append(candidate_input[world[i].item()][idxs[i]].squeeze(dim = 0).tolist())
cand_list = torch.FloatTensor(cand_list)
print(cand_list.shape)