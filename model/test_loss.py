import torch
from config.train_config import train_dataset
from model.loss import MultitaskYOLOLoss
from model.model import MultitaskYOLO
from data.dataloader import DataLoader


if __name__ == "__main__":
    dataset = train_dataset
    dataloader = DataLoader(dataset, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    m = MultitaskYOLO(dataset.class_grouping, dataset.anchors).to(device)
    loss_fn = MultitaskYOLOLoss(dataset.classes, dataset.class_grouping, dataset.anchors).to(device)

    x = torch.randn(1, 3, 640, 640).to(device)

    y = m(x)

    labels = dataloader[0].label_batch
    loss = loss_fn(y, labels)
    for group_name, loss in loss.items():
        print(f"{group_name}:")
        for k, v in loss.items():
            print(f"\t{k} {v}")
    