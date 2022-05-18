
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
# len(data['train'][0])
for i in data['train']:
    print(sorted(data['train'][i]))
    break
    # print(len(data['train'][i]))
# %%
print(data.keys())
print(data['umap'])