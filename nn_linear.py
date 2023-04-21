import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataLoader = DataLoader(dataset, 64)

class Zlb(nn.Module):
    def __init__(self):
        super(Zlb, self).__init__()
        # 线性层需要设置输入的特征数目和输出的特征数目
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


zlb = Zlb()
for data in dataLoader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # torch.flatten可以直接将形状摊平，直接变成一个一维的tensor
    output = torch.flatten(imgs)
    print(output.shape)
    output = zlb(output)
    print(output.shape)