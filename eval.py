import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from config.train_config import *
from data.dataloader import DataLoader
from matplotlib import pyplot as plt
from logger.logger import log_precision_recall
from model.model import MultitaskYOLO, MultitaskYOLOLoss, get_detections
from PIL import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from util.device import device
from util.types import Resolution
from visualization.visualization import get_labeled_img, unpad_labels
from torchvision import transforms


def eval(
    model: MultitaskYOLO,
    epoch: Optional[int],
    writer: Optional[SummaryWriter],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Args:
        model (MultitaskYOLO): -
        epoch (Optional[int]): -
        run_id (Optional[str]): -

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
    if writer is None: # not running training
        run_id=f"eval_{datetime.now().strftime('%Y_%h_%d_%H_%M_%S')}"
        os.makedirs(f"./runs/{run_id}")
        writer = SummaryWriter(f"./runs/{run_id}")

    dataloader = DataLoader(eval_dataset, eval_batch_size, shuffle=False)
    loss_fn = MultitaskYOLOLoss(eval_dataset.classes, eval_dataset.class_grouping, eval_dataset.anchors)
    class_groups = eval_dataset.class_grouping.groups

    pred_results = {
        group_name: [[] for _ in group_classes]
        for group_name, group_classes in class_groups.items()}
    n_gt = {
        group_name: [0 for _ in group_classes]
        for group_name, group_classes in class_groups.items()
    }

    total_losses = {
        "total": 0,
        "xy": 0,
        "wh": 0,
        "conf": 0,
        "cls": 0,
    }
    
    for i in tqdm(range(len(dataloader)), colour="blue", desc="Eval"):
        yolo_input = dataloader[i]
        ids, x, labels = yolo_input.id_batch, yolo_input.image_batch.to(device), yolo_input.label_batch        
        y = model.to(device)(x)     # {"gp_1": [ys, ym, yl]_1, ..., "gp_n": [ys, ym, yl]_n}
        losses, preds = loss_fn(    # {"gp_1": (loss, preds)_1, ..., "gp_n": (loss, preds)_n} loss_data_i
            y,                      # is the i-th group's loss data over all three pred scales
            labels,
            eval=True,
            writer=writer if i == 0 else None,
            epoch=epoch
        )
        for group_name in class_groups.keys():
            for class_idx in range(len(class_groups[group_name])):
                class_mask = losses[group_name]["pred_results"][:, 0] == class_idx
                pred_results[group_name][class_idx].append(losses[group_name]["pred_results"][class_mask][:,1:])
                n_gt[group_name][class_idx] += losses[group_name]["n_gt"][class_idx]
        
        losses = {
            key: sum([group_losses[key] for group_losses in losses.values()])
            for key in total_losses.keys()
        }

        for key in total_losses.keys():
            total_losses[key] += losses[key] / len(dataloader)

        if i == 0 and epoch % visualization_interval == 0:
            detections = get_detections(preds, 0.5, eval_dataset.classes, eval_dataset.class_grouping)
            for b, (id, _, image_detections) in enumerate(zip(ids, x, detections)):
                original_image = Image.open(f"{eval_dataset.root_path}/{id}.jpg")
                image_detections = unpad_labels(Resolution.from_image(original_image), image_detections)
                labeled_image = get_labeled_img(original_image, image_detections, eval_dataset.classes, scale=1.5)
                writer.add_image(f"ep_{epoch}_eval_detections", transforms.ToTensor()(labeled_image), b)
    
    # print(n_gt)

    kpis = {}
    for group_name, classes in class_groups.items():
        for i, class_name in enumerate(tqdm(classes, colour="blue", desc=f"{group_name} APs")):
            class_ap_data = average_precision(
                torch.cat(pred_results[group_name][i]),
                n_gt[group_name][i],
            )

            kpis[f"{class_name} AP"] = class_ap_data["ap"]
            
            if epoch % visualization_interval == 0:
                log_precision_recall(
                    class_ap_data["recall_values"],
                    class_ap_data["precision_values"],
                    group_name,
                    class_name,
                    epoch,
                    writer,
                    i,
                )

    m_ap = torch.mean(torch.tensor([ap for ap in kpis.values() if ap is not torch.nan])).item()
    kpis["mAP"] = m_ap

    return kpis, total_losses


def average_precision(pred_results: Tensor, n_gt: int) -> Dict:
    """calculates AP of one class based on ranked predictions (prediction results)

    Args:
        pred_results (Tensor): tensor of shape (n_pred, 2): [[score, tp/fp], ...]
            (tp/fp is 1 if prediction is a TP 0 if it is a FP).
        n_gt (int): number of ground truth objects

    Returns:
        Dict: 
            {
                "ap": AP wrt. the given class (nan if there are no gt objects) as float
                "recall_values": recall values in a Tensor of shape (n_pred + 1,) ([0] if there are
                    no gt objects)
                "precision_values": interpolated precision values in a Tensor of shape
                    (n_pred + 1,) ([1] if there are no gt objects)
            }

    """
    recall_values = torch.zeros(1).to(device)
    precision_values = torch.ones(1).to(device)

    if n_gt == 0:
        return {
            "ap": torch.nan,
            "recall_values": recall_values,
            "precision_values": precision_values,
        }

    sorted_pred_results = pred_results[pred_results[:, 0].sort(descending=True)[1]]
    recall_values = sorted_pred_results[:, 1].cumsum(0) / n_gt
    precision_values = sorted_pred_results[:, 1].cumsum(0) / (torch.arange(len(pred_results)).to(device) + 1)

    recall_values = torch.cat([torch.zeros(1).to(device), recall_values])
    precision_values = torch.cat([torch.ones(1).to(device), precision_values])

    for i in reversed(range(1, len(precision_values))):
        if precision_values[i - 1] < precision_values[i]:
            precision_values[i - 1] = precision_values[i]

    ap = torch.trapezoid(y=precision_values, x=recall_values).item()

    return {
        "ap": ap,
        "recall_values": recall_values,
        "precision_values": precision_values,
    }
