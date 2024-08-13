import torch
import torch.nn as nn
from torch.functional import F
class TopkLoss(nn.Module):
    def __init__(self, num_class, topk,size_average=True):
        super(TopkLoss, self).__init__()
        self.num_class = num_class
        self.topk = topk
        self.size_average = size_average

    def forward(self, classes, labels):
        classes = F.softmax(classes,dim=-1)
        labels = F.one_hot(labels, self.num_class).float()
        loss = -labels*torch.log(classes)
        # loss = (labels - classes)**2
        loss = loss.sum(dim=-1)
        id = torch.topk(loss,self.topk)[1]
        loss[id] = loss[id]*0

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()