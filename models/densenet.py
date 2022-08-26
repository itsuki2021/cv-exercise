from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, dim_in, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm2d(dim_in))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(dim_in, bn_size * growth_rate,
                                           kernel_size=(1, 1), stride=(1, 1),
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor):
        features = super().forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return torch.concat([x, features], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, dim_in, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(dim_in=dim_in + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(dim_in))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(dim_in, dim_out, kernel_size=(1, 1),
                                          stride=(1, 1), bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, compression_rate=0.5, drop_rate=0,
                 num_classes=1000):
        super().__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features * compression_rate))
                self.features.add_module("transition%d" % (i+1), transition)
                num_features = int(num_features * compression_rate)

        # final bn + ReLU
        self.features.add_module("norms5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """

        :param x:   Tensor, Bx3x224x224
        :return:    Tensor, Bx(num_classes)
        """
        features = self.features(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
