import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bottleneck_width: float):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return torch.cat([x, out1, out2], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bottleneck_width: float):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        if b1.size()[2:] != b2.size()[2:]:
            b2 = F.interpolate(b2, size=b1.size()[2:], mode='bilinear', align_corners=False)

        out = torch.cat([b1, b2], 1)
        out = self.conv2(out)
        return out


class PeleeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, growth_rate: int = 32,
                 block_config: List[int] = [3, 4, 8, 6],
                 bottleneck_width: List[float] = [1, 2, 4, 4]):
        super(PeleeNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('stem', StemBlock(3, 32))

        curr_channels = 32
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f'denseblock{i + 1}',
                                     DenseBlock(num_layers, curr_channels, growth_rate, bottleneck_width[i]))
            curr_channels += num_layers * growth_rate

            self.features.add_module(f'transition{i + 1}',
                                     nn.Sequential(
                                         nn.Conv2d(curr_channels, curr_channels, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(curr_channels),
                                         nn.ReLU(inplace=True)
                                     ))

            if i != len(block_config) - 1:
                self.features.add_module(f'transition{i + 1}_pool',
                                         nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))

        self.classifier = nn.Linear(curr_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def build_peleenet(num_classes: int = 1000) -> PeleeNet:
    return PeleeNet(num_classes=num_classes)