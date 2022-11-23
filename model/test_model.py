from data.datasets.datasets import *
from model.model import MultitaskYOLO
import torch
from torch import Tensor
from config.train_config import train_dataset
import torchinfo
from util.device import device

if __name__ == "__main__":
    dataset = train_dataset
    m = MultitaskYOLO(dataset.class_grouping, dataset.anchors).to(device)
    # torchinfo.summary(m, (3, 416, 416), batch_dim=0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1, device="cpu")

    x = torch.randn(1, 3, 640, 640).to(device)
    y = m(x)

    for group_name, [ys, ym, yl] in y.items():
        print(group_name)
        for y in [ys, ym, yl]:
            print(f"\t{y.size()}")
