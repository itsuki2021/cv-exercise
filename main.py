import torch

from models import MobileNetV2

if __name__ == '__main__':
    size = (2, 3, 224, 224)     # N x C x H x W
    num_classes = 1000

    x = torch.rand(size=size)
    model = MobileNetV2(num_classes=num_classes)
    y = model(x)
    print(y.size())
