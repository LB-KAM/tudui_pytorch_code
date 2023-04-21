# 主要是了解nn.module的用法，我们自己定义一个模型，需要去重写init方法和forward方法，其中forward主要是对数据进行一些加工处理，然后对数据进行一个输出
import torch
from torch import nn


class Zlb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


zlb = Zlb()
x = torch.tensor(1.0)
output = zlb(x)
print(output)