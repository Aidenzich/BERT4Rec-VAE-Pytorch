#%%
import pytorch_lightning as pl
import datasets.ml_1m
from options import args
from dataloaders import dataloader_factory
from utils import *
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from models.bert_modules.bert import BERT
import torch.nn.functional as F
from trainers.utils import recalls_and_ndcgs_for_ks

# %%
new = datasets.ml_1m.ML1MDataset(args)
train_loader, val_loader, test_loader = dataloader_factory(args)

class BertModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
        
    
    def forward(self, x):
        x = self.bert(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        seqs, labels = batch
        logits = self.forward(seqs)  # B x T x V (128 x 100 x 3707) (BATCH x SEQENCE_LEN x ITEM_NUM)        
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = F.cross_entropy(logits, labels, ignore_index=0)
        self.log("train_loss", loss)
        
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)        
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        seqs, candidates, labels = batch
        scores = self.forward(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, [1])
        self.log("recall", metrics)
        
#%% 
Bert = BertModel(args)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(Bert, train_loader, val_loader)

# %%



# %%
