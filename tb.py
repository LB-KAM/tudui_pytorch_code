from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# tensorboard主要是用来显示train_loss，将这些东西可视化
# 可以在命令行当中使用 tensorboard --logdir=[日志路径]

path = "data/train/bees_image/98391118_bdb1e80cce.jpg"
img_PIL = Image.open(path)
img_array = np.array(img_PIL)
print(type(img_array))

# 指定日志存储路径路径
writer = SummaryWriter("logs")
# 第一个参数是标题名称，第二个参数是图像的数据，它必须是tesor类型或者numpy类型或者字符串，第三个参数是数据格式
writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    # 相当于加上标题，y轴，x轴
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
