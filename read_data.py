from torch.utils.data import Dataset
from PIL import Image
import os


# 继承父类Dataset
class MyData(Dataset):
    # 初始化，我们需要给定一个训练集的根路径，以及一个标签路径
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    # 继承了父类Dataset后必须要重写的一个方法之一
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        # img_item_path = os.path.join(self.path, img_name)
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    # 继承了父类Dataset后必须要重写的一个方法之一
    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants"
# 相当于是一个数组，每一个元素是由img和label组成，所以可以用两个变量来接收其中一个数组元素
ants_dataset = MyData(root_dir, ants_label_dir)
img, label = ants_dataset[2]
img.show()
print(label)
bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)
# 可以将数据集拼接起来，当真实的数据集不够用的时候，我们可以使用人造的数据集，将其合成,这里只是把两个数据集拼起来了
# train_dataset = ants_dataset + bees_dataset
