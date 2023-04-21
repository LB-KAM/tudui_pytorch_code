import torch.nn
from torch.nn import L1Loss, MSELoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)
# reduction默认是'mean'，表示做差之后求平均，如果设置为'sum'表示只是把差值的绝对值进行了一个求和
input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss()
output = loss(input, target)

mse_loss = MSELoss()
output_mse = mse_loss(input, target)

print(output)
print(output_mse)


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = torch.nn.CrossEntropyLoss()
output_corss = loss_cross(x, y)
print(output_corss)