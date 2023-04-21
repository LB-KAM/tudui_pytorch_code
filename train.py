import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 神经网络模型单独保存在一个名为model.py的文件当中
from model import *


# 加载训练数据集和测试数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


# 训练集和测试集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# python的格式化输出
print("训练集的长度为:{}".format(train_data_size))
print("测试集的长度为:{}".format(test_data_size))

# 利用DataLoader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建模型实例
zlb = Zlb()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(zlb.parameters(), lr=learning_rate)

# 设置一些网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tesnorBoard
writer = SummaryWriter("./train_logs")

for i in range(epoch):
    print("-------------------第{}轮训练开始-------------------".format(i+1))
    # 很多模型会在训练开始之前使用train()模式 比如zlb.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = zlb(imgs)
        # 利用交叉熵计算损失
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        # 这里loss建议写成loss.item()，因为直接输出loss的话他是一个tensor数据类型，输出loss.item()的话会是直接一个数字,具体测试可以看item_test.py
        if total_train_step % 100 == 0:
            print("训练次数:{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(),total_train_step)


    # 很多模型会在计算损失函数之前调用eval()函数
    zlb.eval()
    # 每一轮训练整体数据集上的损失
    total_test_loss = 0
    # 计算整体正确率
    total_accuracy = 0.0
    # 我们需要判断模型训练后是否会满足我们的预期，因此，我们会在每一轮训练结束后进行测试，我们需要使用测试数据集来评估我们模型的能力,在这里，我们不需要对模型进行调优，所以，我们会在开头写上with torch.no_grad()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zlb(imgs)
            # 这里是在部分测试数据集上的一个损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 方向选择为横向：column
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size)
    total_test_step += 1

    # 通常我们会去保存每一个epoch的模型
    torch.save(zlb, "zlb_{}.pth".format(i))
    print("模型已保存")
writer.close()