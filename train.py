import os
from typing import Dict, Tuple
from model.model import MultitaskYOLO, MultitaskYOLOLoss
from config.train_config import *
from data.dataloader import DataLoader
from util.device import device
from datetime import datetime
import torch
from tqdm import tqdm
from eval import eval
from torch.utils.tensorboard import SummaryWriter
from logger.logger import *
from torch.optim import Optimizer


def train_one_epoch(
    model: MultitaskYOLO,
    epoch: int,
    optimizer: Optimizer,
    writer: SummaryWriter
) -> Dict[str, float]:
    """
    Runs one epoch of training and evals if the epoch is an eval epoch (as per the eval
    interval).

    Returns:
        Dict[str, float]: 
            {
                "total": 0,
                "xy": 0,
                "wh": 0,
                "conf": 0,
                "cls": 0,
            }
    """
    model.train()

    dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True, p_hflip=0.4)
    loss_fn: MultitaskYOLOLoss = MultitaskYOLOLoss(
        train_dataset.classes,
        train_dataset.class_grouping,
        train_dataset.anchors,
    ).to(device)

    total_losses = {
        "total": 0,
        "xy": 0,
        "wh": 0,
        "conf": 0,
        "cls": 0,
    }

    for i in tqdm(range(len(dataloader)), colour="green", desc=f"Train epoch {epoch}"):
        yolo_input = dataloader[i]
        x, labels = yolo_input.image_batch.to(device), yolo_input.label_batch

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(x)

        # Log heatmaps if first batch (w and e get passed to loss)
        w = writer if i == 0 and epoch % visualization_interval == 0 else None
        e = epoch if i == 0 and epoch % visualization_interval == 0 else None

        # Compute loss and gradients
        losses, _ = loss_fn(y, labels, writer=w, epoch=e)
        losses = {
            key: sum([group_losses[key] for group_losses in losses.values()])
            for key in total_losses.keys()
        }
        losses["total"].backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and log
        for key in total_losses.keys():
            total_losses[key] += losses[key] / len(dataloader)

    return total_losses


def eval_one_epoch(
    model: MultitaskYOLO,
    epoch: int,
    writer: SummaryWriter
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Args:
        model (MultitaskYOLO):
        epoch (int):
        run_id (str):

    Returns:
        Tuple[Dict[str, float], Dict[str, Tensor]]: (kpis, losses)
            kpis: {
                "mAP": mAP as float,
                "AP class_1": AP of class 1,
                ...
                "AP class_n": AP of class n,
            }
            (The keys of each class AP is "{class_name} AP")

            losses: {
                "total": ...
                "xy": ...
                "wh": ...
                "conf": ...
                "cls": ...
            }
    """
    model.eval()
    with torch.no_grad():
        kpis, losses = eval(model, epoch, writer)
    
    # text_string = ""
    # for k, v in kpis.items():
    #     text_string += f"{k}: {v}\n"
    # writer.add_text(f"ep_{epoch}_eval_kpis", text_string)

    return kpis, losses


def train(model: MultitaskYOLO, num_epochs: int,  writer: SummaryWriter) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    best_mAP = -1
    for epoch in range(1, num_epochs + 1):
        # train one epoch
        train_losses = train_one_epoch(model, epoch, optimizer, writer)
        log_losses(train_losses, "train", epoch, writer)
        
        # evaluate one epoch if at the end of interval
        if epoch % eval_interval == 0 and epoch >= first_eval_epoch:
            kpis, val_losses = eval_one_epoch(model, epoch, writer)
            log_losses(val_losses, "val", epoch, writer)
            log_kpis(kpis, "eval", epoch, writer)

            # save model if there if a new best
            if kpis["mAP"] > best_mAP:
                best_mAP = kpis["mAP"]
                torch.save(model.state_dict(), f"{writer.log_dir}/saved_models/ep_{epoch}.pt")



if __name__ == "__main__":
    run_id=f"train_{datetime.now().strftime('%Y_%h_%d_%H_%M_%S')}"
    os.makedirs(f"./runs/{run_id}/saved_models")
    writer = SummaryWriter(f"./runs/{run_id}")
    model = MultitaskYOLO(
        train_dataset.class_grouping,
        train_dataset.anchors
    ).to(device)
    if saved_model_path is not None:
        model.load_state_dict(torch.load(saved_model_path))

    train(model, num_epochs, writer)

    writer.close()
