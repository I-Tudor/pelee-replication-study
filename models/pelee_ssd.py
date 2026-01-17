import torch
import torch.nn as nn
from itertools import product
from math import sqrt


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        inter_channels = in_channels // 2
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.res(x) + self.skip(x)


class PeleeSSD(nn.Module):
    def __init__(self, backbone, num_classes, cfg):
        super(PeleeSSD, self).__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.is_backbone = True

        self.res_heads = nn.ModuleList([
            ResBlock(512), ResBlock(704), ResBlock(256),
            ResBlock(256), ResBlock(256)
        ])

        self.exts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(704, 128, 1, bias=False), nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1, bias=False), nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1, bias=False), nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=1, padding=0, bias=False), nn.ReLU(True)
            )
        ])

        num_priors = [len(ar) + 1 for ar in cfg['aspect_ratios']]
        self.loc = nn.ModuleList([nn.Conv2d(256, n * 4, 1) for n in num_priors])
        self.conf = nn.ModuleList([nn.Conv2d(256, n * num_classes, 1) for n in num_priors])


        self.ds_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),  # f1
            nn.Conv2d(512, num_classes, 1)  # f2
        ])

        self.priors = self._get_priors()

    def _get_priors(self):
        priors = []
        img_size = self.cfg['min_dim']
        for i, f_size in enumerate(self.cfg['feature_maps']):
            for i_j, j_i in product(range(f_size), repeat=2):
                cx = (j_i + 0.5) * self.cfg['steps'][i] / img_size
                cy = (i_j + 0.5) * self.cfg['steps'][i] / img_size

                s_k = self.cfg['scales'][i]
                for ar in self.cfg['aspect_ratios'][i]:
                    priors.append([cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)])
                    if ar == 1:
                        s_k_prime = sqrt(s_k * (self.cfg['scales'][i + 1] if i + 1 < len(self.cfg['scales']) else 1.0))
                        priors.append([cx, cy, s_k_prime, s_k_prime])
        return torch.Tensor(priors)

    def forward(self, x):
        stages = self.backbone(x)
        f1, f2, f3 = stages

        sources = [
            self.res_heads[0](f2),
            self.res_heads[1](f3)
        ]

        x_ext = f3
        for i, ext in enumerate(self.exts):
            x_ext = ext(x_ext)
            sources.append(self.res_heads[i + 2](x_ext))

        loc, conf = [], []
        for i, src in enumerate(sources):
            loc.append(self.loc[i](src).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf[i](src).permute(0, 2, 3, 1).contiguous())

        l = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        c = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.training:
            return (
                l.view(l.size(0), -1, 4),
                c.view(c.size(0), -1, self.num_classes),
                self.priors,
                self.ds_heads[0](f1),
                self.ds_heads[1](f2)
            )
        return l.view(l.size(0), -1, 4), c.view(c.size(0), -1, self.num_classes), self.priors