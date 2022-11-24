from typing import Dict, List, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from abc import ABC, abstractmethod

from config.config import model_input_resolution as img_res
from data.dataset import ObjectLabel, ClassGrouping
from model.common import get_anchor_masks
from visualization.visualization import visualize_heatmap
from util.device import device
from util.nms import nms
import config.config as cfg


class MeanLossBase(ABC):
    @abstractmethod
    def concrete_loss_fn(input: Tensor, target: Tensor) -> Tensor:
        pass

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        assert(input.size() == target.size())
        if len(input) == 0:
            return torch.tensor(0)
        else:
            return self.concrete_loss_fn(input, target)


class MeanMSELoss(MeanLossBase):
    def concrete_loss_fn(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
    

class MeanBCELoss(MeanLossBase):
    def concrete_loss_fn(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target)


class YOLOLoss(nn.Module):
    """

    Forward method:
        xs (Tensor): Tensor of shape (nb, na_s * (5 + nc), ng_s, ng_s), where na_s and ng_s are
            the number of small anchors, and the number of grid cells in a given row/column for
            small scale predictions respectively, eg. (4, 3 * (5 + nc), 80, 80)
        xm (Tensor): Same as xs but for medium scale predictions; Tensor of shape 
            (nb, na_m * (5 + nc), ng_m, ng_m), eg. (4, 2 * (5 + nc), 40, 40) if ng_s is 80.   
        xl (Tensor): Same as xm but for large scale predictions; Tensor of shape
            (nb, na_l * (5 + nc), ng_l, ng_l), eg. (4, 2 * (5 + nc), 20, 20) if ng_m is 40.
        labels (Optional[List[List[ObjectLabel]]]): Ground truth object labels for every image
            in batch.
        eval (bool): Whether The loss function also needs to return precision recall curve data
        visualize (bool): Whether confidence heatmaps are visualized

    Returns:
        Union[Dict,Tensor]: There are 3 cases.
            1. Training: Dict containing loss values (total, xy, wh, conf and cls)
            2. Eval: Dict containing loss values (total, xy, wh, conf and cls) and ranked
                predictions. The latter is in the form of a tensor of shape (num_pred, 2)
                (confidence and correctness for each prediction).
            3. Inference: Tensor in the shape (nb, num_pred, nch). Here, nch is ()
    """
    def __init__(
        self,
        all_classes: List[str],
        class_group: List[str],
        all_anchors: List[List[int]],
        anchor_masks: List[List[int]]
    ) -> None:
        super().__init__()
        self.device = device

        self.strides = [8, 16, 32]

        self.class_indices = [all_classes.index(class_name) for class_name in class_group]
        self.nc = len(class_group)
        self.all_anchors = all_anchors
        self.anchor_masks = anchor_masks

        self.ignore_thres = 0.5
        self.lambda_noobj = 1
        self.lambda_obj = 10
        self.lambda_coord = 1

        self.bce_loss = MeanBCELoss()
        self.mse_loss = MeanMSELoss()
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for stride, mask in zip(self.strides, self.anchor_masks):
            all_anchors_grid = torch.tensor([(w / stride, h / stride) for w, h in self.all_anchors]).to(self.device)
            ref_anchors = torch.cat([
                torch.zeros_like(all_anchors_grid).to(self.device),
                all_anchors_grid,
            ], dim=1)
            masked_anchors = all_anchors_grid[mask]

            na = len(mask)
            ng = img_res.w // stride
            grid_x = torch.arange(ng, dtype=torch.float).repeat(na, ng, 1).to(self.device)
            grid_y = torch.arange(ng, dtype=torch.float).repeat(na, ng, 1).permute(0, 2, 1).to(self.device)
            anchor_w = masked_anchors[:, 0].repeat(ng, ng, 1).permute(2, 0, 1).to(self.device)
            anchor_h = masked_anchors[:, 1].repeat(ng, ng, 1).permute(2, 0, 1).to(self.device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(
        self,
        pred: Tensor,
        labels: List[List[ObjectLabel]],
        nch: int,
        output_idx: int,
        visualize = False
    ) -> Tuple:
        stride = self.strides[output_idx]
        anchor_mask = self.anchor_masks[output_idx]
        nb, na, ng, ng, nch = pred.size()

        masked_anchors = self.masked_anchors[output_idx]            # Tensor (na, 2)
        ref_anchors = self.ref_anchors[output_idx]                  # Tensor (na_all, 4)

        obj_mask = torch.zeros(nb, na, ng, ng)
        noobj_mask = torch.ones(nb, na, ng, ng)
        target = torch.zeros(nb, na, ng, ng, nch).to(self.device)

        n_gt = [0 for _ in self.class_indices]
        pred_results = torch.cat([
            pred[..., 5:].argmax(-1).type(torch.float).view(nb, na, ng, ng, 1),     # cls
            pred[..., 4].view(nb, na, ng, ng, 1),                                   # score
            torch.zeros(nb, na, ng, ng).to(device).view(nb, na, ng, ng, 1)          # tp/fp
        ], dim=-1)
        for b in range(nb):
            obj_labels = [l for l in labels[b] if l.cls in self.class_indices]
            n = len(obj_labels)
            if n == 0:
                continue

            gt = torch.tensor([
                l.bbox
                for l
                in obj_labels]
            ).to(self.device) * img_res.w / stride                          # (n, 4)
            ref_gt = torch.cat([                                            # (n, 4)
                torch.zeros(n, 2).to(self.device),
                gt[:, 2:]
            ], dim=1)
            gt_i = gt[:, 0].type(torch.int)                                 # (n)
            gt_j = gt[:, 1].type(torch.int)                                 # (n)

            anchor_ious_all = box_iou(ref_gt, ref_anchors)                  # (n, na_all)
            anchor_ious_masked = anchor_ious_all[:, anchor_mask]            # (n, na)

            best_n_all = anchor_ious_all.argmax(dim = 1)                    # (n)
            best_n_mask = torch.isin(                                       # (n)
                best_n_all,
                torch.tensor(anchor_mask).to(self.device)
            )

            for i, l in enumerate(obj_labels):
                if best_n_mask[i]:
                    n_gt[self.class_indices.index(l.cls)] += 1
            
            n = best_n_mask.sum().item()
            if n == 0:
                continue

            for obj_i, gt_box in enumerate(gt):
                if best_n_mask[obj_i]:
                    i, j = gt_i[obj_i], gt_j[obj_i]
                    a = best_n_all[obj_i] % len(anchor_mask)

                    obj_mask[b, a, j, i] = 1
                    noobj_mask[b, a, j, i] = 0
                    noobj_mask[b, anchor_ious_masked[obj_i] > self.ignore_thres, i, j] = 0

                    target[b, a, j, i, 0:2] = gt_box[0:2] - torch.floor(gt_box[0:2])
                    target[b, a, j, i, 2:4] = torch.log(gt_box[2:4] / masked_anchors[a] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + self.class_indices.index(obj_labels[obj_i].cls)] = 1

                    iou = box_iou(
                        pred[b, a, j, i, :4].unsqueeze(0),
                        box_convert(gt[obj_i].unsqueeze(0), "cxcywh", "xyxy"),
                    )

                    pred_score = pred[b, a, j, i, 4]
                    pred_class = torch.argmax(pred[b, a, j, i, 5:])
                    target_class = self.class_indices.index(obj_labels[obj_i].cls)

                    # if iou > cfg.eval_iou_match_threshold and pred_score > cfg.detection_score_threshold and pred_class == target_class:
                    #     pred_results[b, a, j, i, 2] = 1
                    pred_results[b, a, j, i, 2] = 1

        pred_results = pred_results.view(-1, 3)
        pred_results = pred_results[pred_results[:, 1] > cfg.detection_score_threshold]

        if visualize:
            visualize_heatmap(target, pred, output_idx)

        return obj_mask, noobj_mask, target, n_gt, pred_results

    def forward(
        self,
        ys: Tensor,
        ym: Tensor,
        yl: Tensor,
        labels = None,
        eval = False,
        visualize = False
    ) -> Union[Dict,Tensor]:
        loss_xy = torch.zeros(3)
        loss_wh = torch.zeros(3)
        loss_conf = torch.zeros(3)
        loss_cls = torch.zeros(3)
        n_gt_total = [0 for _ in self.class_indices]
        preds: List[Tensor] = []
        outputs: List[Tensor] = []
        for idx, output in enumerate([ys, ym, yl]):
            nb = output.size(0)
            na = len(self.anchor_masks[idx])
            ng = output.size(2)

            grid_x = self.grid_x[idx].expand(nb, -1, -1, -1)     # Tensor (nb, na, ng, ng)
            grid_y = self.grid_y[idx].expand(nb, -1, -1, -1)     # Tensor (nb, na, ng, ng)
            anchor_w = self.anchor_w[idx].expand(nb, -1, -1, -1) # Tensor (nb, na, ng, ng)
            anchor_h = self.anchor_h[idx].expand(nb, -1, -1, -1) # Tensor (nb, na, ng, ng)

            # [nb, na * (5 + nc), ng, ng] -> [nb, na, ng, ng, (5 + nc)]
            output = output.view(nb, na, 5 + self.nc, ng, ng)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(output)

            # logistic activation for xy, obj, cls
            xy_obj_cls_mask = [0, 1] + [i for i in range(4, 5 + self.nc)]
            output[..., xy_obj_cls_mask] = torch.sigmoid(output[..., xy_obj_cls_mask])

            pred = output.clone()
            pred[..., 0] += grid_x
            pred[..., 1] += grid_y
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h
            preds.append(pred)
        
        if eval or labels is None:
            nms(*preds, cfg.nms_iou_threshold)

        preds_image_space = [None for _ in preds]
        pred_results = [None for _ in preds]
        for idx, (output, pred) in enumerate(zip(outputs, preds)):
            if labels is not None:
                obj_mask, noobj_mask, target, n_gt, pred_results[idx] = self.build_target(
                    pred = torch.detach(pred),
                    labels = labels,
                    nch = 5 + self.nc,
                    output_idx = idx,
                    visualize = visualize
                )
                for i in range(len(self.class_indices)):
                    n_gt_total[i] += n_gt[i]

                obj_mask = obj_mask.type(torch.ByteTensor).bool()
                noobj_mask = noobj_mask.type(torch.ByteTensor).bool()
                
                # x and y loss
                loss_xy[idx] = self.lambda_coord * self.mse_loss(
                    input = output[..., :2][obj_mask],
                    target = target[..., :2][obj_mask]
                )

                # width and height loss
                loss_wh[idx] = self.lambda_coord * self.mse_loss(
                    input = output[..., 2:4][obj_mask],
                    target = target[..., 2:4][obj_mask]
                )
                
                # confidence loss
                loss_obj = self.bce_loss(
                    input = output[..., 4][obj_mask],
                    target = target[..., 4][obj_mask]
                )
                loss_noobj = self.bce_loss(
                    input = output[..., 4][noobj_mask],
                    target = target[..., 4][noobj_mask]
                )
                loss_conf[idx] = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
                
                # classification loss
                loss_cls[idx] = self.bce_loss(
                    input = output[..., 5:][obj_mask],
                    target = target[..., 5:][obj_mask]
                )
            else:
                pred[..., :4] = pred[..., :4] * self.strides[idx]
                preds_image_space[idx] = pred.view(pred.size(0), -1, pred.size(-1))

        if labels is not None:
            loss = loss_xy.sum() + loss_wh.sum() + loss_conf.sum() + loss_cls.sum()
            return {
                "total": loss,
                "xy": loss_xy.sum(),
                "wh": loss_wh.sum(),
                "conf": loss_conf.sum(),
                "cls": loss_cls.sum(),
                "n_gt": n_gt_total,                     # List: i-th element is number of gts in the i-th class
                "pred_results": torch.cat(pred_results) # Tensor (n, 3): [[class, score, tp/fp], ...]
            }
        else:
            return torch.cat(preds_image_space, dim=1)

class MultitaskYOLOLoss(nn.Module):
    """
    Constructor args:
        all_classes (List[str]): All all_classes of a dataset
        class_grouping (ClassGrouping): Class grouping of a dataset
        anchors (Dict[str, List[List[float]]]): Anchors for each class group where each entry's key
            is a class group's name and the value is a list of anchors ([[w, h], ..., [w, h]])
    
    Forward method args:
        mt_output (Dict[str, List[Tensor]]): [ys, ym, yl] for each class group
        labels (Optional[List[List[ObjectLabel]]]): A list of object labels for each image in batch
            If this is None, no actual loss will be calculated; predictions will be returned instead   
    """
    def __init__(
        self,
        all_classes: List[str],
        class_grouping: ClassGrouping,
        anchors: Dict[str, List[List[int]]],
    ) -> None:
        super().__init__()
        self.class_grouping = class_grouping
        self.loss_layers = nn.ModuleDict({
            group_name: YOLOLoss(
                all_classes,
                class_group,
                group_anchors,
                get_anchor_masks(group_anchors)
            )
            for (group_name, class_group), group_anchors
            in zip(
                class_grouping.groups.items(),
                anchors.values(),
            )
        })
    
    def forward(
        self,
        mt_output: Dict[str, List[Tensor]],
        labels: Optional[List[List[ObjectLabel]]] = None,
        eval = False,
    ) -> Dict[str, Tuple]:
        return {
            group_name: self.loss_layers[group_name](
                *mt_output[group_name],
                labels=labels,
                eval=eval
            )
            for group_name
            in self.class_grouping.groups.keys()
        }
