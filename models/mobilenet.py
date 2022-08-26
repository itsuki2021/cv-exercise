import torch
from torch import Tensor
from typing import Union, Tuple, Optional
import torch.nn as nn


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.add_module("conv2d", nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups,
                                            bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        self.add_module("relu", nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]],
                 expand_ratio: int):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 point-wise conv
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))

        layers.extend([
            # 3x3 depth-wise conv
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            # 1x1 point-wise conv
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super().__init__()
        input_channels = make_divisible(32 * alpha, round_nearest)
        last_channels = make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # conv1 layer
        features = [ConvBNReLU(3, input_channels, stride=2)]
        # building inverted residual residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channels = make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, output_channels, stride, expand_ratio=t))
                input_channels = output_channels
        # building last several layers
        features.append(ConvBNReLU(input_channels, last_channels, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        N, _, _, _ = x.size()
        x = self.features(x)
        x = self.avg_pool(x).view(N, -1)
        x = self.classifier(x)
        return x
