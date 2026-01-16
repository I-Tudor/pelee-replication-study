import torch
import torch.nn as nn
from itertools import product
from math import sqrt
from typing import List, Tuple


class ResBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + self.skip(x)


class PeleeSSD(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, cfg: dict):
        super(PeleeSSD, self).__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.backbone = backbone.features

        self.res_heads = nn.ModuleList([
            ResBlock(512),
            ResBlock(704),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256)
        ])

        self.ext1 = nn.Sequential(
            nn.Conv2d(704, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ext2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ext3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.loc = nn.ModuleList([nn.Conv2d(256, 6 * 4, kernel_size=1) for _ in range(5)])
        self.conf = nn.ModuleList([nn.Conv2d(256, 6 * num_classes, kernel_size=1) for _ in range(5)])
        self.priors = self._get_priors()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_priors(self) -> torch.Tensor:
        priors = []
        for i, f_size in enumerate(self.cfg['feature_maps']):
            for i_j, j_i in product(range(f_size), repeat=2):
                f_k = self.cfg['steps'][i]
                cx, cy = (j_i + 0.5) / f_k, (i_j + 0.5) / f_k
                s_k = self.cfg['scales'][i]
                for ar in self.cfg['aspect_ratios'][i]:
                    priors.append([cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)])
                    if ar == 1:
                        s_k_prime = sqrt(s_k * (self.cfg['scales'][i + 1] if i + 1 < len(self.cfg['scales']) else 1.0))
                        priors.append([cx, cy, s_k_prime, s_k_prime])

        return torch.clamp(torch.Tensor(priors), 0.0, 1.0)

    def forward(self, x: torch.Tensor):
        sources = []
        curr_x = x
        stage3_feat = None

        for i, layer in enumerate(self.backbone):
            curr_x = layer(curr_x)
            if i == 8:
                stage3_feat = curr_x

        sources.append(self.res_heads[0](stage3_feat))
        sources.append(self.res_heads[1](curr_x))

        x_ext = self.ext1(curr_x)
        sources.append(self.res_heads[2](x_ext))
        x_ext = self.ext2(x_ext)
        sources.append(self.res_heads[3](x_ext))
        x_ext = self.ext3(x_ext)
        sources.append(self.res_heads[4](x_ext))

        loc_preds, conf_preds = [], []
        for i, src in enumerate(sources):
            loc_preds.append(self.loc[i](src).permute(0, 2, 3, 1).contiguous())
            conf_preds.append(self.conf[i](src).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)
        return (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors.to(x.device))