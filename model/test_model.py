from data.datasets.datasets import *
from model.model import YOLOv5Net
import torch
import torchinfo

if __name__ == "__main__":
    dataset = PascalVOCDataset(dataset_type="train", shuffle=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YOLOv5Net(dataset.classes)
    # x = torch.randn(1, 3, 640, 640).to(device)
    # y = m(x)
    # print(y)
    torchinfo.summary(m, (3, 640, 640), batch_dim=0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1, device="cpu")
