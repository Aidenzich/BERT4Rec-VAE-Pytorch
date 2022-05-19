
#%% 
import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from datasets import dataset_factory
from trainers import trainer_factory
from utils import *

# %%
dataset = dataset_factory(args)
data = dataset.load_dataset()

# %%
from tqdm import tqdm

for i in data['train']:
    print(sorted(data['train'][i]))
    print(data['test'])
    break



# %%
from models.bert import BERTModel
export_root = setup_train(args)
train_loader, val_loader, test_loader = dataloader_factory(args)
model = BERTModel(args)



# %%
best_model = torch.load(os.path.join('experiments/test_2022-05-18_18/', 'models', 'best_acc_model.pth')).get('model_state_dict')
model.load_state_dict(best_model)
model.eval()

#%%
k = 10
with torch.no_grad():
    tqdm_dataloader = tqdm(val_loader)
    i = 0
    for batch_idx, batch in enumerate(tqdm_dataloader):
        i+=1
        # batch = [x.to('cpu') for x in batch]
        # candiates 與 seqs 不會交集
        # candiates 第一位是正確答案
        seqs, candidates, labels = batch        
        scores = model(seqs)                    # B x T x V                        
        scores = scores[:, -1, :]               # B x V
        scores = scores.gather(1, candidates)   # B x C
        print(scores.shape)
        print(candidates.shape)
        print(scores)
        # 如果要 inference，用 scores 與 candidates，取 score 最大者
        
        
        
        labels_float = labels.float()           # 每個 labels 都一樣
        rank = (-scores).argsort(dim=1)         # return index, 由大至小
        C_uni = np.unique(candidates.numpy()).squeeze()
        S_uni = np.unique(seqs.numpy()).squeeze()
        # print(np.intersect1d(C_uni, S_uni))
        
        # print(rank)
        # print(labels_float)
        # hits = labels_float.gather(1, rank)     
        # print(hits)
        # print("\n","\n")
        
        
        # if i > 1:
        break
    
        # hits 找排名第 0 者        
        # print()
        # print(rank)
        # print(hits)
        # rank = rank[:, :k]
        # print()
        # print(rank)
        # hits = labels_float.gather(1, rank)
        # print()
        # print(hits)
        # break                        
        # print(scores.shape)                
        # print(scores[0])
        # print(rank[0])
        # print(labels.shape)
        # labels_float = labels.float()
        # hits = labels_float.gather(1, rank)
        # print(hits[0])
        # print(labels.sum(1))
        # print(labels.shape)
        # print((hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item())


# %%
