from typing import Dict, Optional
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


def log_precision_recall(
    recall_values: Tensor,
    precision_values: Tensor,
    group_name: str,
    class_name: str,
    epoch: Optional[int]=None,
    writer: Optional[SummaryWriter]=None,
    step: Optional[int]=None
) -> None:
    
    data = torch.cat([recall_values.view(-1, 1), precision_values.view(-1, 1)], dim=1).cpu().tolist()

    text_string = ""
    for pair in data:
        text_string += f"{pair[0]} {pair[1]}"
    writer.add_text(f"ep_{epoch}_{class_name}_pr_data", text_string)
    # np.savetxt(
    #     f"{log_dir}/{class_name}_prcurves_data.txt",
    #     torch.cat([recall_values.view(-1, 1), precision_values.view(-1, 1)], dim=1).cpu().numpy(),
    #     "%1.9f",
    # )

    fig, ax = plt.subplots()
    fig.suptitle(class_name)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.plot(recall_values.tolist(), precision_values.tolist())
    # fig.savefig(f"{log_dir}/{class_name}_prcurves.png")
    writer.add_figure(f"ep_{epoch}_{group_name}_pr_curves", fig, step, close=True)
    plt.close(fig)
