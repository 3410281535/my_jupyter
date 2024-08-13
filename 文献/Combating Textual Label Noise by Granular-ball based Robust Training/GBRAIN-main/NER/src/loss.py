import torch
import torch.nn.functional as F
from torch import nn


# Generalized Cross Entropy Loss
class GCELoss(nn.Module):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
             
    def forward(self, logits, targets, sb):
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        if logits.size(-1) == 1:
           #print(targets)
           ce_loss = nn.BCEWithLogitsLoss(reduction='none')
           loss = ce_loss(logits.view(-1), targets.float())
        else:
           ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
           loss = ce_loss(logits, targets)
        
        return loss
        
        
        
