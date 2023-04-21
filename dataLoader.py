import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
test_Loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# 测试数据集当中的第一张图片大小和目标
img, target = test_dataset[0]
print(img.shape)
print(target)

# 通过这种形式我们可以按照批量大小去读取测试数据集，其中DataLoader会将批量当中的图片一起打包，也会将target一起打包
writer = SummaryWriter("dataLoader")

for epoch in range(2):
    step = 0
    for data in test_Loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch:{}".format(epoch), imgs, step)
        step += 1


writer.close()