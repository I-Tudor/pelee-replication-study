import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_width):
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

    def forward(self, x):
        return torch.cat([x, self.branch1(x), self.branch2(x)], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck_width):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck_width)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def forward(self, x):
        x = self.conv1(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        if b1.size()[2:] != b2.size()[2:]:
            b2 = F.interpolate(b2, size=b1.size()[2:], mode='bilinear', align_corners=False)
        return self.conv2(torch.cat([b1, b2], 1))


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pooling=True):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_pooling:
            x = self.pool(x)
        return x


class PeleeNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=[3, 4, 8, 6], bottleneck_width=[1, 2, 4, 4]):
        super(PeleeNet, self).__init__()

        self.stem = StemBlock(3, 32)
        curr_channels = 32

        self.stage1_dense = DenseBlock(block_config[0], curr_channels, growth_rate, bottleneck_width[0])
        curr_channels += block_config[0] * growth_rate
        self.stage1_trans = TransitionBlock(curr_channels, curr_channels)

        self.stage2_dense = DenseBlock(block_config[1], curr_channels, growth_rate, bottleneck_width[1])
        curr_channels += block_config[1] * growth_rate
        self.stage2_trans = TransitionBlock(curr_channels, curr_channels)

        self.stage3_dense = DenseBlock(block_config[2], curr_channels, growth_rate, bottleneck_width[2])
        curr_channels += block_config[2] * growth_rate
        self.stage3_trans = TransitionBlock(curr_channels, curr_channels)

        self.stage4_dense = DenseBlock(block_config[3], curr_channels, growth_rate, bottleneck_width[3])
        curr_channels += block_config[3] * growth_rate
        self.stage4_trans = TransitionBlock(curr_channels, curr_channels,
                                            use_pooling=False)  # No pooling in last transition

        self.num_classes = num_classes

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1_dense(x)
        x = self.stage1_trans(x)

        x = self.stage2_dense(x)
        f1 = x
        x = self.stage2_trans(x)

        x = self.stage3_dense(x)
        f2 = x
        x = self.stage3_trans(x)

        x = self.stage4_dense(x)
        x = self.stage4_trans(x)
        f3 = x

        return [f1, f2, f3]


def build_peleenet(num_classes=1000, pretrained_path='pretrain/peleenet.pth'):
    model = PeleeNet(num_classes=num_classes)
    if pretrained_path:
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        except:
            pass
    return model