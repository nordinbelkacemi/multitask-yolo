from model.model import MultitaskYOLO, MultitaskYOLOLoss
from data.dataloader import DataLoader
from config.train_config import eval_dataset, batch_size
from util.device import device
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def eval(model: MultitaskYOLO, epoch: int, dataset=eval_dataset):
    dataloader = DataLoader(dataset, batch_size)
    loss_fn = MultitaskYOLOLoss(dataset.classes, dataset.class_grouping, dataset.anchors)
    class_groups = dataset.class_grouping.groups

    pred_results = {
        group_name: [[] for _ in class_groups[group_name]]
        for group_name in class_groups.keys()}
    n_gt = {
        group_name: [0 for _ in class_groups[group_name]]
        for group_name in class_groups.keys()
    }

    for i in tqdm(range(len(dataloader))):
        yolo_input = dataloader[i]
        _, x, labels = yolo_input.id_batch, yolo_input.image_batch.to(device), yolo_input.label_batch
        y = model.to(device)(x)                 # {"gp_1": [ys, ym, yl]_1, ..., "gp_n": [ys, ym, yl]_n}
        loss = loss_fn(y, labels, eval=True)    # {"gp_1": loss_data_1, ..., "gp_n": loss_data_n} loss_data_i
                                                # is the i-th group's loss data over all three pred scales
        for group_name in class_groups.keys():
            for i in range(len(class_groups[group_name])):
                class_mask = loss[group_name]["pred_results"][:, 0] == i
                pred_results[group_name][i].append(loss[group_name]["pred_results"][class_mask][:,1:])
                n_gt[group_name][i] += loss[group_name]["n_gt"][i]

    print(n_gt)

    aps = {}
    for group_name, classes in class_groups.items():
        print(f"{group_name}...")
        for i, class_name in enumerate(tqdm(classes)):
            aps[class_name] = average_precision(
                torch.cat(pred_results[group_name][i]),
                n_gt[group_name][i],
                class_name,
                epoch,
            )
    
    return aps

def average_precision(pred_results: Tensor, n_gt: int, class_name: str, epoch: int) -> float:
    sorted_pred_results = pred_results[pred_results[:, 0].sort(descending=True)[1]]
    recall_values = sorted_pred_results[:, 1].cumsum(0) / n_gt
    precision_values = sorted_pred_results[:, 1].cumsum(0) / (torch.arange(len(pred_results)).to(device) + 1)

    recall_values = torch.cat([torch.zeros(1).to(device), recall_values])
    precision_values = torch.cat([torch.ones(1).to(device), precision_values])

    corrected_precision_values = precision_values.clone()
    for i in reversed(range(1, len(corrected_precision_values))):
        if corrected_precision_values[i - 1] < corrected_precision_values[i]:
            corrected_precision_values[i - 1] = corrected_precision_values[i]
    
    np.savetxt(
        f"./out/ep_{epoch}_{class_name}_prcurves_data.txt",
        torch.cat([recall_values.view(-1, 1), precision_values.view(-1, 1)], dim=1).cpu().numpy(),
        "%1.9f",
    )
    np.savetxt(
        f"./out/ep_{epoch}_{class_name}_prcurves_data_corrected.txt",
        torch.cat([recall_values.view(-1, 1), corrected_precision_values.view(-1, 1)], dim=1).cpu().numpy(),
        "%1.9f",
    )

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    axs[0].set_xlabel("recall")
    axs[0].set_ylabel("precision")
    axs[0].plot(recall_values.tolist(), precision_values.tolist())
    axs[1].set_xlabel("recall")
    axs[1].set_ylabel("precision")
    axs[1].plot(recall_values.tolist(), corrected_precision_values.tolist())
    fig.savefig(f"./out/ep_{epoch}_{class_name}_prcurves.png")

    return torch.trapezoid(y=corrected_precision_values, x=recall_values)
