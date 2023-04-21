import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten


device = torch.device("cuda:0")
path = "./images/dog.jpg"

img = Image.open(path).convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

# transform = torchvision.transforms.Resize((32, 32))

img = transform(img)
print(img.shape)
class Zlb(nn.Module):
    def __init__(self):
        super(Zlb, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("./zlb_39.pth")
print(model)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.to(device)
model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(1).item())