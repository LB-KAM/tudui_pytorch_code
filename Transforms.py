from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# Transforms.ToTensor的用法
path = 'dataset/train/ants/0013035.jpg'
img = Image.open(path)
tensor_trans = transforms.ToTensor()
# Tensor数据类型，主要是包含了反向神经网络所需要的一些参数
tensor_img = tensor_trans(img)
# print(tensor_img)
writer = SummaryWriter("logs")
writer.add_image("tensor_img", tensor_img)
writer.close()