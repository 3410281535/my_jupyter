import torch  
import torch.nn.functional as F  
  
def triplet_loss(anchor, positive, negative, margin=0.5):  
    """  
    计算三元组损失  
    anchor: 锚点样本  
    positive: 正样本  
    negative: 负样本  
    margin: 距离差阈值  
    """  
    dist_positive = F.pairwise_distance(anchor, positive)  
    dist_negative = F.pairwise_distance(anchor, negative)  
    loss = torch.max(torch.zeros_like(dist_positive), dist_positive - dist_negative + margin)  
    return loss.mean()

# anchor = torch.randn(100, 128)  # 锚点样本，大小为(100, 128)  
# positive = torch.randn(100, 128)  # 正样本，大小为(100, 128)  
# negative = torch.randn(100, 128)  # 负样本，大小为(100, 128)  
# loss = triplet_loss(anchor, positive, negative, margin=0.5)  # 计算三元组损失

a = torch.tensor([1,1,2])
b = torch.tensor([[1,1,4],[1,1,4]])

print(F.pairwise_distance(a, b) )