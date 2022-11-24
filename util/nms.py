from typing import List
import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_convert

def nms(ys: Tensor, ym: Tensor, yl: Tensor, iou_threshold: float) -> None:
    """
    Runs non maximum suppression on predictions at 3 scales: First, predictions from the 3
    different scales are merged, scores below 0.5 are filtered out and nms is run using
    torchvision.ops.batched_nms. Boxes that are not kept remain in place, but their score is
    changed to 0.

    Args:
        ys (Tensor): Tensor of shape (nb, na_s, ng_s, ng_s, 5 + nc)
        ym (Tensor): Tensor of shape (nb, na_m, ng_m, ng_m, 5 + nc)
        yl (Tensor): Tensor of shape (nb, na_l, ng_l, ng_l, 5 + nc)
    
    Returns:
        List[Tensor]: [ys, ym, yl] with 0 score at predictions that were not kept by nms
    """
    ns, nm, nl = (y.size(1) * y.size(2) * y.size(3) for y in [ys, ym, yl])

    boxes_batch = torch.cat([
        y.view(y.size(0), -1, y.size(-1))[:, :, :4]
        for y
        in [ys, ym, yl]
    ], dim=1)
    scores_batch = torch.cat([
        y.view(y.size(0), -1, y.size(-1))[:, :, 4]
        for y
        in [ys, ym, yl]
    ], dim=1)
    idxs_batch = torch.cat([
        y.view(y.size(0), -1, y.size(-1))[:, :, 5:].argmax(dim=-1)
        for y
        in [ys, ym, yl]
    ], dim=1)
    score_mask_batch = scores_batch > 0.5
    
    for i, (boxes, scores, idxs, score_mask) in enumerate(zip(boxes_batch, scores_batch, idxs_batch, score_mask_batch)):
        rem_indices = batched_nms(
            boxes=box_convert(boxes[score_mask], "cxcywh", "xyxy"),
            scores=scores[score_mask],
            idxs=idxs[score_mask],
            iou_threshold=iou_threshold,
        )
        rem_mask = score_mask.nonzero(as_tuple=True)[0][rem_indices]

        rem_mask_s = rem_mask[torch.logical_and(rem_mask >= 0, rem_mask < ns).nonzero(as_tuple=True)]
        rem_mask_m = rem_mask[torch.logical_and(rem_mask >= ns, rem_mask < ns + nm).nonzero(as_tuple=True)]
        rem_mask_l = rem_mask[torch.logical_and(rem_mask >= ns + nm, rem_mask < ns + nm + nl).nonzero(as_tuple=True)]

        rem_mask_m = rem_mask_m % ns
        rem_mask_l = rem_mask_l % (ns + nm)

        for y, mask in zip([ys, ym, yl], [rem_mask_s, rem_mask_m, rem_mask_l]):
            rem_confs = y.view(y.size(0), -1, y.size(-1))[i, :, 4][mask]
            y.view(y.size(0), -1, y.size(-1))[i, :, 4] = 0            
            y.view(y.size(0), -1, y.size(-1))[i, :, 4][mask] = rem_confs
