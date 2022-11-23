import torch
from config.train_config import train_dataset
from model.loss import MultitaskYOLOLoss
from model.model import MultitaskYOLO
from data.dataloader import DataLoader
from util.device import device
import datetime


if __name__ == "__main__":
    dataset = train_dataset
    dataloader = DataLoader(dataset, 1)

    m = MultitaskYOLO(dataset.class_grouping, dataset.anchors).to(device)
    loss_fn = MultitaskYOLOLoss(dataset.classes, dataset.class_grouping, dataset.anchors).to(device)

    x = torch.randn(1, 3, 640, 640).to(device)
    y = m(x)

    labels = dataloader[0].label_batch

    # Loss calculation
    loss = loss_fn(y, labels)
    for group_name, loss in loss.items():
        print(f"{group_name}:")
        for k, v in loss.items():
            print(f"\t{k} {v}")

    # Prediction and visualization
    timestamp = int(datetime.datetime.now().timestamp())
    loss = loss_fn(y)
    
