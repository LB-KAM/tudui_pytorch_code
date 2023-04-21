import torch
import torch.nn.functional as f

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernal = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 对于conv2d函数，输入的tensor维度需要修改，需要输入bath_size channel，H和W
input = torch.reshape(input, (1, 1, 5, 5))
kernal = torch.reshape(kernal, (1, 1, 3, 3))

print(input.shape)
print(kernal.shape)

# 输入是输入数据，weight是指卷积核，stride是指移动的步长
output = f.conv2d(input=input, weight=kernal, stride=1)
print(output)
# 不同的步长输出的结果不一样，数据的维度也不一样
output = f.conv2d(input=input, weight=kernal, stride=2)
print(output)

# padding主要是在图像的周围进行一个填充，如果padding=1，那么就是在图像的周围进行了一个一格子的放大，增大的格子当中的值默认的情况下是0
output = f.conv2d(input=input, weight=kernal, stride=1, padding=1)
print(output)