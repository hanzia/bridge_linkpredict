import torch
import torch.tensor as tensor


a = torch.ones((2,4)).float()  # norm仅支持floatTensor,a是一个2*4的Tensor
a0 = torch.norm(a, p=1, dim=0)  # 按0维度求2范数
a1 = torch.norm(a, p=1, dim=1)  # 按1维度求2范数
b = torch.randn(2000,300)
print(b)