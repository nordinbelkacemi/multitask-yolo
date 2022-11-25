import os
from model.model import MultitaskYOLO, MultitaskYOLOLoss
from config.train_config import train_dataset, eval_dataset, batch_size, num_epochs, lr
from data.dataloader import DataLoader
from util.device import device
from datetime import datetime
import torch
from tqdm import tqdm
from eval import eval


def train_one_epoch(model: MultitaskYOLO, epoch: int, run_id: str,) -> None:
    """
    Runs one epoch of training and evals if the epoch is an eval epoch (as per the eval
    interval).
    """
    model.train()

    dataloader = DataLoader(train_dataset, batch_size)
    loss_fn = MultitaskYOLOLoss(
        train_dataset.classes,
        train_dataset.class_grouping,
        train_dataset.anchors
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    total_losses = {
        "total": 0,
        "xy": 0,
        "wh": 0,
        "conf": 0,
        "cls": 0,
    }

    for i in tqdm(range(4), colour="green", desc=f"Epoch {epoch}"):
        yolo_input = dataloader[i]
        x, labels = yolo_input.image_batch.to(device), yolo_input.label_batch

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(x)

        # Compute loss and gradients
        losses = loss_fn(y, labels)
        losses = {
            key: sum([group_losses[key] for group_losses in losses.values()])
            for key in total_losses.keys()
        }
        losses["total"].backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and log
        for key in total_losses.keys():
            total_losses[key] += losses[key]


def eval_one_epoch(model: MultitaskYOLO, epoch: int, run_id: str) -> None:
    model.eval()
    eval(model, epoch, run_id)


def train(model: MultitaskYOLO, num_epochs: int, run_id: str) -> None:
    os.mkdir(f"./runs/{run_id}")

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, epoch, run_id)
        eval_one_epoch(model, epoch, run_id)


if __name__ == "__main__":
    train(
        model=MultitaskYOLO(
            train_dataset.class_grouping,
            train_dataset.anchors
        ).to(device),
        num_epochs=num_epochs,
        run_id=f"train_{datetime.now().strftime('%Y_%h_%d_%H_%M_%S')}",
    )
