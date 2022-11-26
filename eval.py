from typing import Dict, Optional, Tuple
from model.model import MultitaskYOLO, MultitaskYOLOLoss
from data.dataloader import DataLoader
from config.train_config import eval_dataset, batch_size
from util.device import device
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime


def eval(
    model: MultitaskYOLO,
    epoch: Optional[int],
    run_id: Optional[str],
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
    if run_id is None:
        run_id = f"eval_{datetime.now().strftime('%Y_%h_%d_%H_%M_%S')}"
        os.mkdir(f"./runs/{run_id}")

    dataloader = DataLoader(eval_dataset, batch_size)
    loss_fn = MultitaskYOLOLoss(eval_dataset.classes, eval_dataset.class_grouping, eval_dataset.anchors)
    class_groups = eval_dataset.class_grouping.groups

    pred_results = {
        group_name: [[] for _ in class_groups[group_name]]
        for group_name in class_groups.keys()}
    n_gt = {
        group_name: [0 for _ in class_groups[group_name]]
        for group_name in class_groups.keys()
    }

    total_losses = {
        "total": 0,
        "xy": 0,
        "wh": 0,
        "conf": 0,
        "cls": 0,
    }

    num_batches = 4

    for i in tqdm(range(num_batches), colour="blue", desc="Eval"):
        yolo_input = dataloader[i]
        _, x, labels = yolo_input.id_batch, yolo_input.image_batch.to(device), yolo_input.label_batch
        y = model.to(device)(x)                 # {"gp_1": [ys, ym, yl]_1, ..., "gp_n": [ys, ym, yl]_n}
        losses = loss_fn(y, labels, eval=True)    # {"gp_1": loss_data_1, ..., "gp_n": loss_data_n} loss_data_i
                                                # is the i-th group's loss data over all three pred scales
        for group_name in class_groups.keys():
            for i in range(len(class_groups[group_name])):
                class_mask = losses[group_name]["pred_results"][:, 0] == i
                pred_results[group_name][i].append(losses[group_name]["pred_results"][class_mask][:,1:])
                n_gt[group_name][i] += losses[group_name]["n_gt"][i]
        
        losses = {
            key: sum([group_losses[key] for group_losses in losses.values()])
            for key in total_losses.keys()
        }

        for key in total_losses.keys():
            total_losses[key] += losses[key] / num_batches
    
    # print(n_gt)

    kpis = {}
    for group_name, classes in class_groups.items():
        for i, class_name in enumerate(tqdm(classes, colour="blue", desc=group_name)):
            class_ap_data = average_precision(
                torch.cat(pred_results[group_name][i]),
                n_gt[group_name][i],
            )

            kpis[f"{class_name} AP"] = class_ap_data["ap"]
            log_precision_recall(
                class_ap_data["recall_values"],
                class_ap_data["precision_values"],
                class_name,
                epoch,
                run_id,
            )

    m_ap = torch.mean(torch.tensor([ap for ap in kpis.values() if ap is not torch.nan])).item()
    kpis["mAP"] = m_ap

    with open(f"./runs/{run_id}/kpis.txt", "w") as f:
        for k, v in kpis.items():
            f.write(f"{k}: {v}\n")

    return kpis, total_losses


def average_precision(pred_results: Tensor, n_gt: int) -> Dict:
    """calculates AP of one class based on ranked predictions (prediction results)

    Args:
        pred_results (Tensor): tensor of shape (n_pred, 2): [[score, tp/fp], ...]
            (tp/fp is 1 if prediction is a TP 0 if it is a  FP).
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


def log_precision_recall(recall_values: Tensor, precision_values: Tensor, class_name: str, epoch: Optional[int], run_id: str):
    if epoch is not None:
        epoch_prefix = f"ep_{epoch}_"
    else:
        epoch_prefix = ""

    np.savetxt(
        f"./runs/{run_id}/{epoch_prefix}{class_name}_prcurves_data.txt",
        torch.cat([recall_values.view(-1, 1),
                  precision_values.view(-1, 1)], dim=1).cpu().numpy(),
        "%1.9f",
    )

    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.plot(recall_values.tolist(), precision_values.tolist())
    fig.savefig(f"./runs/{run_id}/{epoch_prefix}{class_name}_prcurves.png")
    plt.close(fig)
