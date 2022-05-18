
#%% 
import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from datasets import dataset_factory
from trainers import trainer_factory
from utils import *

#%%
# train_loader, val_loader, test_loader = dataloader_factory(args)
# %%
dataset = dataset_factory(args)
data = dataset.load_dataset()

# %%
from tqdm import tqdm

for i in data['train']:
    print(sorted(data['train'][i]))
    break
    # print(len(data['train'][i]))
#%%
print(data['test'])
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
with torch.no_grad():
    tqdm_dataloader = tqdm(val_loader)
    for batch_idx, batch in enumerate(tqdm_dataloader):
        # k = 10
        # batch = [x.to('cpu') for x in batch]
        seqs, candidates, labels = batch
        # scores = model(seqs)  # B x T x V
        # print(scores.shape)
        # scores = scores[:, -1, :]  # B x V
        # print(scores.shape)
        # scores = scores.gather(1, candidates)  # B x C
        # print(scores.shape)
        # rank = (-scores).argsort(dim=1)
        # rank = rank[:, :k]
        # print(scores[0])
        # print(rank[0])
        # print(labels.shape)
        # labels_float = labels.float()
        # hits = labels_float.gather(1, rank)
        # print(hits[0])
        # print(labels.sum(1))
        print(labels.shape)
        # print((hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item())

