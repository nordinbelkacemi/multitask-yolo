from typing import List
from model.common import *
import config.config as cfg
import torch.nn as nn
from data.dataset import Dataset

class BackBone(nn.Module):
    def __init__(
        self,
        first_ch_out=cfg.mod_feat_0
    ) -> None:
        super().__init__()
        self.d1 = ConvBNSiLU(ch_in=3, ch_out=first_ch_out, k=6, s=2, p=2)                           # 64
        self.d2 = ConvBNSiLU(ch_in=self.d1.ch_out, ch_out=self.d1.ch_out * 2, k=3, s=2, p=1)        # 128
        self.c3_1 = C3(ch_in=self.d2.ch_out, ch_out=self.d2.ch_out, n=2, shortcut=True, e=0.5)      # 128
        self.d3 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out * 2, k=3, s=2, p=1)    # 256
        self.c3_2 = C3(ch_in=self.d3.ch_out, ch_out=self.d3.ch_out, n=4, shortcut=True, e=0.5)      # 256
        self.d4 = ConvBNSiLU(ch_in=self.c3_2.ch_out, ch_out=self.c3_2.ch_out * 2, k=3, s=2, p=1)    # 512
        self.c3_3 = C3(ch_in=self.d4.ch_out, ch_out=self.d4.ch_out, n=6, shortcut=True, e=0.5)      # 512
        self.d5 = ConvBNSiLU(ch_in=self.c3_3.ch_out, ch_out=self.c3_3.ch_out * 2, k=3, s=2, p=1)    # 1024
        self.c3_4 = C3(ch_in=self.d5.ch_out, ch_out=self.d5.ch_out, n=2, shortcut=True, e=0.5)      # 1024
    
    def forward(self, x):
        x1 = self.c3_2(self.d3(self.c3_1(self.d2(self.d1(x)))))
        x2 = self.c3_3(self.d4(x1))
        x3 = self.c3_4(self.d5(x2))
        return [x1, x2, x3]
    
class Neck(nn.Module):
    def __init__(
        self,
        ch_in=cfg.mod_feat_0 * 16,
        first_ch_out=cfg.mod_feat_0 * 16,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.sppf = SPPF(ch_in=ch_in, ch_out=first_ch_out)
        self.cv1 = ConvBNSiLU(ch_in=self.sppf.ch_out, ch_out=self.sppf.ch_out // 2, k=1, s=1, p=0)
        self.c3_1 = C3(ch_in=self.cv1.ch_out * 2, ch_out=self.cv1.ch_out, n=2, shortcut=False, e=1.0)
        self.cv2 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out // 2, k=1, s=1, p=0)
        self.c3_2 = C3(ch_in=self.cv2.ch_out * 2, ch_out=self.cv2.ch_out, n=2, shortcut=False, e=1.0)
        
    def forward(self, x1, x2, x3):
        x4 = self.cv1(self.sppf(x3))
        x5 = self.cv2(self.c3_1(torch.cat([x2, self.up(x4)], dim=1)))
        x6 = self.c3_2(torch.cat([x1, self.up(x5)], dim=1))
        return [x6, x5, x4]
    
class Head(nn.Module):
    def __init__(
        self,
        ch_in=cfg.mod_feat_0 * 4,
        first_ch_out=cfg.mod_feat_0 * 4
    ) -> None:
        super().__init__()
        self.cv1 = ConvBNSiLU(ch_in=ch_in, ch_out=first_ch_out, k=3, s=2, p=1)
        self.c3_1 = C3(ch_in=self.cv1.ch_out * 2, ch_out=self.cv1.ch_out * 2, n=2, shortcut=False, e=0.5)
        self.cv2 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out, k=3, s=2, p=1)
        self.c3_2 = C3(ch_in=self.cv2.ch_out * 2, ch_out=self.cv2.ch_out * 2, n=2, shortcut=False, e=0.5)
    
    def forward(self, x6, x5, x4):
        x7 = x6
        x8 = self.c3_1(torch.cat([x5, self.cv1(x7)], dim=1))
        x9 = self.c3_2(torch.cat([x4, self.cv2(x8)], dim=1))
        return [x7, x8, x9]
    
class YOLOv5(nn.Module):
    def __init__(self, classes: List[str]):
        super().__init__()
        self.nc = len(classes)

        self.backbone = BackBone()
        self.neck = Neck()
        self.head = Head()

        self.dtr = nn.ModuleList([
            nn.Conv2d(cfg.mod_feat_0 * 4, (5 + self.nc) * 3, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(cfg.mod_feat_0 * 8, (5 + self.nc) * 3, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(cfg.mod_feat_0 * 16, (5 + self.nc) * 3, kernel_size=1, stride=1, padding=0),
        ])

        self.dtrs = [self.dtr for _ in range(20)]

    def forward(self, x):
        [x7, x8, x9] = self.head(*self.neck(*self.backbone(x)))

        outputs = []
        for dtr in self.dtrs:
            [ys, ym, yl] = [cv(x) for cv, x in zip(dtr, [x7, x8, x9])]
            outputs.append([ys, ym, yl])
        return outputs


class MultitaskYOLO(nn.Module):
    def __init__(self, class_groups: Dict[str, List[str]]),
