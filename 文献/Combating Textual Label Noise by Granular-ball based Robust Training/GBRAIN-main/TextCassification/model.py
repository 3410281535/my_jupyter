import math

import torch
import torch.nn.functional as F
from myrelu import GBNR
from torch import nn
from torch.nn import ReLU
from torch.nn.parameter import Parameter
cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, in_length, out_length,
                 num_class, routing_type, embedding_type, num_repeat, dropout):
        super().__init__()


        self.in_length, self.out_length = in_length, out_length
        self.hidden_size=hidden_size
		# embedding 层是一个很大的矩阵，[vocab_size,embdeeing_size] 这个矩阵是所有的词向量，这些词向量都是参数，会随着梯度更新而更新
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0)
		# 定义lstm层
        self.features = nn.LSTM(embedding_size, self.hidden_size, num_layers=2, dropout=dropout, batch_first=True,
                               bidirectional=True)
		# 定义一个全连接层，输出48维
        # self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)/
       
	    # 定义分类器层，输出4分类
        self.classifier = nn.Linear(in_features=self.hidden_size, out_features=num_class, bias=False)

    def forward(self, x,target,index,original_target,flag,purity=None):
		
		# 输入样本 x[batch_size,len] 是每个单词在词汇表中的id，和embedding中的矩阵对应，输出是词向量[batch_size,len,128] 128是词向量维度
        embed = self.embedding(x)
		
		# 将词向量输入lstm
        out, _ = self.features(embed)
		
		# 将整句话的lstm输出变成一个向量  此时out 大小为 [batch_size,128] #每一个样本，也就是一句话，变成一个词向量
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        out = out.mean(dim=1).contiguous()
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        hidden = out
		# 将隐向量经过一个全连接层 变成 48维
        
        
		
		# 判断是否过粒球层
        _gbId = None
        center_v = None
        if flag == 0:
			# 把输入数据，调成粒球输入的形式
            out = torch.cat((target.reshape(-1, 1), out), dim=1)  
            out = torch.cat((index.reshape(-1, 1), out), dim=1)  
            out = torch.cat((original_target.reshape(-1, 1), out), dim=1)  
            pur_tensor = torch.Tensor([[purity]] * out.size(0))
            out = torch.cat((pur_tensor.to(device), out), dim=1)  
			
            
            sss = out[:,4:]
			# 输入粒球层，聚球，输出 out,是球心向量，target：球心标签，sample_index：需要加入经验池的数据id
            out, target, sample_index, chaifen_id, _gbId = GBNR.apply(out.to(cpu_device))  
            center_v = out
            out, target, sample_index, chaifen_id = out.to(device), target.to(device), sample_index.to(
                device), chaifen_id.to(device)
            sample_index, chaifen_id = sample_index.tolist(), chaifen_id.tolist()
            index = []
            for i in chaifen_id:
                index.append(sample_index[:i])
                sample_index = sample_index[i:]
            # print("index:", index)
            #print("本次加入经验池球数：", len(index))
		
		# 将最后的输出向量经过分类器层
        out = self.classifier(out)

        return out, target, index,hidden,_gbId,center_v

class Modelcnn(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, in_length, out_length,
                 num_class, routing_type, embedding_type, num_repeat, dropout):
        super().__init__()


        self.in_length, self.out_length = in_length, out_length
        self.hidden_size=hidden_size
		# embedding 层是一个很大的矩阵，[vocab_size,embdeeing_size] 这个矩阵是所有的词向量，这些词向量都是参数，会随着梯度更新而更新
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0)
		# 定义lstm层
        filter_sizes = [3,4,5]
        self.features = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=50,
                      kernel_size=(fs, 100))
            for fs in filter_sizes
        ])
        # self.fc = nn.Linear(300,150)
		
        self.classifier = nn.Linear(in_features=self.hidden_size, out_features=num_class, bias=False)

    def forward(self, x,target,index,original_target,flag,purity=None):
		
		# 输入样本 x[batch_size,len] 是每个单词在词汇表中的id，和embedding中的矩阵对应，输出是词向量[batch_size,len,128] 128是词向量维度
        embed = self.embedding(x)
        
        embedded = embed.unsqueeze(1)
        
        out = [F.relu(conv(embedded)).squeeze(3) for conv in self.features]
        
		# 将整句话的lstm输出变成一个向量  此时out 大小为 [batch_size,128] #每一个样本，也就是一句话，变成一个词向量
        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in out]  # [batch,num_filter]
        
        out=torch.cat(pooled, dim=1)
        
        # out = self.fc(out)
        
       
        hidden = out
		
        
        
		
		# 判断是否过粒球层
        _gbId = None
        center_v = None
        if flag == 0:
			# 把输入数据，调成粒球输入的形式
            out = torch.cat((target.reshape(-1, 1), out), dim=1)  
            out = torch.cat((index.reshape(-1, 1), out), dim=1)  
            out = torch.cat((original_target.reshape(-1, 1), out), dim=1)  
            pur_tensor = torch.Tensor([[purity]] * out.size(0))
            out = torch.cat((pur_tensor.to(device), out), dim=1)  
			
            
            sss = out[:,4:]
			# 输入粒球层，聚球，输出 out,是球心向量，target：球心标签，sample_index：需要加入经验池的数据id
            out, target, sample_index, chaifen_id, _gbId = GBNR.apply(out.to(cpu_device))  
            center_v = out
            out, target, sample_index, chaifen_id = out.to(device), target.to(device), sample_index.to(
                device), chaifen_id.to(device)
            sample_index, chaifen_id = sample_index.tolist(), chaifen_id.tolist()
            index = []
            for i in chaifen_id:
                index.append(sample_index[:i])
                sample_index = sample_index[i:]
            # print("index:", index)
            #print("本次加入经验池球数：", len(index))
		
		# 将最后的输出向量经过分类器层
        out = self.classifier(out)

        return out, target, index,hidden,_gbId,center_v
