import torchvision
import torch

vgg16 = torchvision.models.vgg16(False)
# 方式一，保存模型的结构和参数
# 注意：对于方式一而言，我们保存了网络结构后要去读取它，加载处是要知道其网络结构的定义的，要不然会出现报错
torch.save(vgg16, "vgg16_method1.pth")

# 方式二，保存模型的参数(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")