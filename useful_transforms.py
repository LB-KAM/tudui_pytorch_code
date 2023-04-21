from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

path = 'images/cat.png'
img = Image.open(path).convert('RGB')

writer = SummaryWriter("logs")

# ToTensor
img_trans = transforms.ToTensor()
img_tensor = img_trans(img)

# TensorBoard
writer.add_image("ToTensor", img_tensor, 0)

print(img_tensor[0][0][0])
# Normalize 对于每一个通道进行归一化 输入的是标准值和方差
trans_norm = transforms.Normalize([0.5, 3, 6], [1, 2, 8])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 0)

# resize  size (sequence or int) 这里的sequence是指一个序列，比如说（H,W）如果指定了一个int这就表明按照最小的边来进行一个缩放
print(img.size)
trans_resize = transforms.Resize((128, 128))
img_resize = trans_resize(img)
img_resize_tensor = img_trans(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)
print(img_resize)

# compose - resize -2
trans_resize_2 = transforms.Resize(256)

# compose的参数是一个列表，[数据1，数据2，......]，在compose当中需要的数据是transforms类型，数据之间前面的一个输出是后一个的输入
trans_compose = transforms.Compose([trans_resize_2, img_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
# 随机裁剪一个256*256的区域
# trans_random = transforms.RandomCrop(256)
# 随机裁剪指定大小的区域
trans_random = transforms.RandomCrop((128, 128))
trans_compose_2 = transforms.Compose([trans_random, img_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
