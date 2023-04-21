import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

# 对于RuLU函数，需要指定input的一个batsize，所以需要对其进行reshape
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
dataLoader = DataLoader(dataset, batch_size=64)

class Zlb(nn.Module):
    def __init__(self):
        super(Zlb, self).__init__()
        # inplace如果是True,如果这个值是＜0，那么我们直接将其修改到原来的数字上，如果为False的话，我们就不在原来的数字上进行修改，然后返回一个值，一般情况下建议使用False，因为可以保留原始的数据
        self.relu1 = nn.ReLU(inplace=False)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

zlb = Zlb()
# output = zlb(input)
# print(output)
writer = SummaryWriter("nn_sigmoid")

step = 0
for data in dataLoader:
    imgs, targets = data
    inputs = imgs
    writer.add_images("input", imgs, step)
    output = zlb(imgs)
    writer.add_images("output", output, step)
    step += 1


writer.close()