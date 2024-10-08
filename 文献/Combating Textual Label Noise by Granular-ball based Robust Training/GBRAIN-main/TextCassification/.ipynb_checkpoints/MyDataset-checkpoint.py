from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from collections import Counter
import random
import copy
class MyDataset(Dataset):
    def __init__(self,dataset,noise_ratio=None):
        super(MyDataset, self).__init__()
        self.lens = len(dataset)
        label = []
        data = []


        for i in dataset:
            label.append(i["label"].item())
            data.append(np.array(i["text"]))
        pd_data=pd.DataFrame(columns=["label","text"])
        pd_data['label']=label
        pd_data['text']=data
        # print(pd_data.head(10))
        cnt = Counter(pd_data['label'])
        print("各个标签数量：",cnt)
        label_set = list(cnt.keys())

        def noise(org_label, labels):

            while True:
                noise_label = random.sample(labels, 1)[0]
                if noise_label != label:
                    return noise_label

        data_list = []
        for label in label_set:
            data1 = pd_data[pd_data['label'] == label]
            length = int(data1.shape[0] * noise_ratio)
            noise_labels = [noise(label, label_set) for i in range(length)]
            data1["noise_label"]  = data1['label']
            data1['noise_label'][:length] = noise_labels
            data_list.append(data1)
        self.datafram = pd.concat(data_list, axis=0, ignore_index=True)



    def __getitem__(self, idx):
        data = self.datafram['text'][idx]
        data = np.array(data)
        org_label = self.datafram['label'][idx]
        # org_label = torch.LongTensor(org_label)
        noise_label = self.datafram['noise_label'][idx]
        # noise_label = torch.LongTensor(noise_label)

        return data,(int(noise_label),idx),org_label
    def __len__(self):
        return self.lens