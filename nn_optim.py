import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset, batch_size=64)

class Zlb(nn.Module):
    def __init__(self):
        super(Zlb, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


zlb = Zlb()
loss = nn.CrossEntropyLoss()
# 定义优化器
optim = torch.optim.SGD(zlb.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataLoader:
        imgs, targets = data
        outputs = zlb(imgs)
        result_loss = loss(outputs, targets)
        # 首先需要对模型当中的参数进行梯度清零
        optim.zero_grad()
        # 反向传播求梯度
        result_loss.backward()
        # 将模型模型当中的参数进行一个更新
        optim.step()
        running_loss = running_loss + result_loss
    print(result_loss)
