from transformers import RobertaForTokenClassification
from transformers.modeling_roberta import RobertaLMHead
from torch import nn
from myrelu import GBNR
import torch.nn.functional as F
import torch
from collections import Counter
import numpy
import random
cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GBRAINModel(RobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.hidden_size,256)
        self.fc2 = nn.Linear(256,150)
        self.classifier = nn.Linear(150,4)
        self.bin_classifier = nn.Linear(150, 1)
        self.init_weights()
    def forward(self, input_ids, attention_mask, valid_pos,labels=None,flage=0):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        valid_output = sequence_output[valid_pos > 0]
        
        if labels is not None and True:
            labels = labels[valid_pos > 0]
            # 扔掉 -100的标签
            labels = labels.reshape(-1)
            valid_output = valid_output[labels>=0]
            labels = labels[labels>=0]
        valid_output = F.relu(self.fc1(valid_output))
        valid_output = F.relu(self.fc2(valid_output))
        if self.training and False:
            #print(labels)
            labels1 = labels.to(cpu_device).numpy()
            cnt = Counter(labels1)
            tup_ct = cnt.most_common()
            max_label = tup_ct[0][0]
            max_labels_ct =tup_ct[0][1]
            tup_2 = tup_ct[1:]
            sum=0
            for i in tup_2:
              sum+=i[1]
            avg = sum//len(tup_2)
            
            #avg = random.randint(avg,min(sum,max_labels_ct))
            # avg = (sum+max_labels_ct)//2
            # avg = min(sum,max_labels_ct)
            avg = max_labels_ct // 2
            id = numpy.where(labels1==max_label)[0]
            id2 = numpy.where(labels1!=max_label)[0]
            save_id = random.sample(list(id),avg)
            ids = numpy.concatenate([id2,save_id])
            labels = labels[ids]
            #print(labels)
            valid_output = valid_output[ids]
        valid_output = self.dropout(valid_output)
        # 这里加粒球
        if flage==1:
            out = torch.cat((labels.reshape(-1, 1), valid_output.reshape(-1,150)), dim=1)

            pur_tensor = torch.Tensor([[0.8]] * out.size(0))
            out = torch.cat((pur_tensor.to(device), out), dim=1)
            out, target ,sb= GBNR.apply(out.to(cpu_device))
            out, target = out.to(device), target.to(device),
        else:
            out = valid_output
            target = labels
            sb=None

        sequence_output = self.dropout(out)
        logits = self.classifier(sequence_output)
        bin_logits = self.bin_classifier(sequence_output)
        return logits, bin_logits,target,sb


