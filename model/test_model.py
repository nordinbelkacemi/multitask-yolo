from data.datasets.datasets import *
from model.model import MultitaskYOLO
import torch
from torch import Tensor
from config.train_config import train_dataset
from util.device import device
# import torchinfo
from torchsummary import summary

if __name__ == "__main__":
    dataset = train_dataset
    m = MultitaskYOLO(dataset.class_grouping, dataset.anchors).to(device)
    # summary(m, (3, 640, 640))

    x = torch.randn(1, 3, 640, 640).to(device)
    y = m(x)

    for group_name, [ys, ym, yl] in y.items():
        print(group_name)
        for y in [ys, ym, yl]:
            print(f"\t{y.size()}")
