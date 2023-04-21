import torch
import torchvision.models

# 方式1->保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
print(model)
# 由于方式2保存的是模型的参数，所以我们需要先将其恢复成网络模型
vgg16 = torchvision.models.vgg16(False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)