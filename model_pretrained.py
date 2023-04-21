import torchvision
from torch import nn

# 数据集太大，官方已经不支持直接下载了
# dataset = torchvision.datasets.ImageNet("./data_image_net", split='train', download=True,
#                                         transform=torchvision.transforms.ToTensor())

# 默认情况下progress设置为True，显示下载进度
vgg16_false = torchvision.models.vgg16(False)
vgg16_true = torchvision.models.vgg16(True)


# 如果我们想要自己去增加/修改这个模型，想在把它用到我们自己的任务上，VGG16这个模型是1000分类的，如果我们想要十分类，可以在最后加上一个线性层，输出特征数是1000，输出特征是10
# vgg16_false.add_module("add_linear", nn.Linear(1000, 10))
# 在classifier当中加入这个模块
vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))

# 我们想要去修改classifier中编号为6的层
vgg16_true.classifier[6] = nn.Linear(4096, 10)

# 我们可以通过debug的形式看到这两个网络之间的权重参数的不同之处
print(vgg16_false)
print(vgg16_true)
