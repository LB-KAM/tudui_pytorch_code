import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset, batch_size=64)

class Zlb(nn.Module):
    # 这一部分的初始化，说明了我们的网络当中带有一层卷积
    def __init__(self):
        super(Zlb, self).__init__()
        # 这里因为是彩色图像，所以in_channel=3，我们想让out_channel=6，卷积核大小kernal_size=3.就是进行一个3*3的卷积，步长设置为1，填充设置为0
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)


    def forward(self, x):
        x = self.conv1(x)
        return x

zlb = Zlb()
# 直接打印，我们可以直接看到网络的结构
print(zlb)
writer = SummaryWriter("nn_conv_2d")
step = 0
for data in dataLoader:
    imgs, targets = data
    output = zlb(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1

writer.close()