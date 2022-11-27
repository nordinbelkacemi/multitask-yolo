from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import torch
from visualization.visualization import visualize_heatmap
from torch import Tensor
from matplotlib import pyplot as plt


def log_losses(losses: Dict[str, float], sub_tag: str, epoch: int, writer: SummaryWriter) -> None:
    """
    Args:
        losses (Dict[str, float]):
            {
                "total": float
                "xy": ...
                "wh": ...
                "conf": ...
                "cls": ...
            }
        sub_tag (str): "train" or "val"
        epoch (int): epoch number
        writer (SummaryWriter): writer used for logging
    """
    loss_components = ["total", "xy", "wh", "conf", "cls"]
    tag_scalar_dict = {k: v for k, v in losses.items() if k in loss_components}
    writer.add_scalars(
        f"Loss/{sub_tag}",
        tag_scalar_dict,
        epoch,
    )


def log_kpis(kpis: Dict[str, float], sub_tag: str, epoch: int, writer: SummaryWriter) -> None:
    """
    Args:
        kpis (Dict[str, float]):
            {
                "mAP": mAP as float,
                "AP class_1": AP of class 1,
                ...
                "AP class_n": AP of class n,
            }
            (The keys of each class AP is "{class_name} AP")
        sub_tag: "train" or "val"
        epoch (int): epoch number
        writer (SummaryWriter): writer used for logging
    """
    tag_scalar_dict = {"mAP": kpis["mAP"]}
    for k, v in kpis.items():
        if v is not torch.nan:
            tag_scalar_dict[k] = v
    writer.add_scalars(
        f"mAP/{sub_tag}",
        tag_scalar_dict,
        epoch,
    )


def log_heatmap(
    target: Tensor,
    pred: Tensor,
    output_idx: int,
    num_anchors: int,
    eval: bool,
    epoch: int,
    writer: SummaryWriter,
) -> None:
    fig = visualize_heatmap(target, pred, output_idx, num_anchors)
    tag = f"ep_{epoch}_heatmap" if not eval else f"ep_{epoch}_eval_heatmap"
    writer.add_figure(tag, fig, output_idx, close=True)
