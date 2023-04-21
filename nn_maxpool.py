import torch.tensor
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataLoader = DataLoader(dataset, batch_size=64)

# 最大池化的作用，经过这样子的处理，所以在训练的时候所需要计算的参数就变小了，训练的更快，一个更加直观的理解就是，拿视频的清晰度来对比，如果我们将1080p的视频作为一个输入的话，720p是经过处理的结果，但是720p仍然能够保证观看的一个需求，其中的数据量减少了
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


class Zlb(nn.Module):
    def __init__(self):
        super(Zlb, self).__init__()
        # 卷积核的大小如果是单个数字，就生成这个数字为边的正方形的卷积核，ceil_mode=False的时候，如果卷积核超出了图像的边际，采取的是丢弃的策略，如果是ceil的话，仍采用保留的策略
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


zlb = Zlb()
# print(zlb)
# output = zlb(input)
# print(output)

writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = zlb(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()