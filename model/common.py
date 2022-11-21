from typing import List
import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    """
    Constructor params:
        ch_in (int): number of input channels,
        ch_out (int): number of output channels,
        k (int): kernel size
        s (int): stride
        p (int): padding (default is 0)
    
    Forward:
        acitvation(batchnorm(conv(x)))
    """
    def __init__(self, ch_in: int, ch_out: int, k: int, s: int, p: int):
        super().__init__()
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, p)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch, shortcut=True):
        super().__init__()
        self.ch_out = ch
        self.cv1 = ConvBNSiLU(ch_in=ch, ch_out=ch, k=1, s=1, p=0)
        self.cv2 = ConvBNSiLU(ch_in=ch, ch_out=ch, k=3, s=1, p=1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.ch_out = ch_out
        ch_ = int(ch_out * e)  # hidden channels
        self.cv1 = ConvBNSiLU(ch_in=ch_in, ch_out=ch_, k=1, s=1, p=0)
        self.cv2 = ConvBNSiLU(ch_in=ch_in, ch_out=ch_, k=1, s=1, p=0)
        self.cv3 = ConvBNSiLU(ch_in=2 * ch_, ch_out=ch_out, k=1, s=1, p=0)
        self.m = nn.Sequential(*(Bottleneck(ch_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, ch_in, ch_out, k=5):
        super().__init__()
        self.ch_out = ch_out
        c_ = ch_in // 2  # hidden channels
        self.cv1 = ConvBNSiLU(ch_in=ch_in, ch_out=c_, k=1, s=1, p=0)
        self.cv2 = ConvBNSiLU(ch_in=c_ * 4, ch_out=ch_out, k=1, s=1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def get_anchor_masks(anchors: List[List[float]]) -> List[List[int]]:
    result = [[], [], []]
    for i in range(len(anchors)):
        result[i % 3].append(i)
    return result
