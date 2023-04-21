import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 通过下列代码的控制台输出我们可以发现，这个CIFAR10数据集当中的每一张图片都是PIL的Image类型，而放入pytorch当中使用，我们需要将其转换为tensor数据类型，这里就需要用torchvision中的Transforms
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# # classes和class_to_idx是一个class和具体数据的对应
# print(test_set.classes)
# print(test_set.class_to_idx)
#
# # 通过打印这个test_set[0]我们可以发现，这个元素由两部分组成，分别是class和target，我们可以通过打断点看到测试集里面的具体组成
# print(test_set[0])
# img, target = test_set[0]
# # test_set[0]的具体内容显示
# img.show()
# print(target)
#
# # 我们可以通过classes[target]来确定具体的类别,通过输出，我们可以看到target==3时，对应的类别是cat
# print(test_set.classes[target])

print(test_set[0])
writer = SummaryWriter("cifarlog")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
