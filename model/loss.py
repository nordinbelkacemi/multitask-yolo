from typing import Dict, List, Callable, Tuple

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
    def __init__(self, classes: List[str], class_group: List[str], all_anchors: List[List[int]], anchor_masks: List[List[int]]):
        super().__init__()
        self.strides = [8, 16, 32]

        self.class_indices = [classes.index(class_name) for class_name in class_group]
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
            all_anchors_grid = torch.tensor([(w / stride, h / stride) for w, h in self.all_anchors])
            ref_anchors = torch.cat([
                torch.zeros_like(all_anchors_grid),
                all_anchors_grid,
            ], dim=1)
            masked_anchors = all_anchors_grid[mask]

            na = len(mask)
            ng = img_res.w // stride
            grid_x = torch.arange(ng, dtype=torch.float).repeat(na, ng, 1)
            grid_y = torch.arange(ng, dtype=torch.float).repeat(na, ng, 1).permute(0, 2, 1)
            anchor_w = masked_anchors[:, 0].repeat(ng, ng, 1).permute(2, 0, 1)
            anchor_h = masked_anchors[:, 1].repeat(ng, ng, 1).permute(2, 0, 1)

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
        target = torch.zeros(nb, na, ng, ng, nch)

        n_matched_objs = 0
        n_correct = 0
        for b in range(nb):
            obj_labels = [l for l in labels[b] if l.cls in self.class_indices]
            n = len(obj_labels)
            if n == 0:
                continue

            gt = torch.tensor([l.bbox for l in obj_labels]) * img_res.w / stride    # (n, 4)
            ref_gt = torch.cat([                                                    # (n, 4)
                torch.zeros(n, 2),
                gt[:, 2:]
            ], dim=1)
            gt_i = gt[:, 0].type(torch.int)                                         # (n)
            gt_j = gt[:, 1].type(torch.int)                                         # (n)

            anchor_ious_all = box_iou(ref_gt, ref_anchors)                          # (n, na_all)
            anchor_ious_masked = anchor_ious_all[:, anchor_mask]                    # (n, na)

            best_n_all = anchor_ious_all.argmax(dim = 1)                            # (n)
            best_n_mask = torch.isin(best_n_all, torch.tensor(anchor_mask))         # (n)
            
            n = best_n_mask.sum().item()
            n_matched_objs += n
            if n == 0:
                continue

            for obj_i, gt_box in enumerate(gt):
                if best_n_mask[obj_i]:
                    i, j = gt_i[obj_i], gt_j[obj_i]
                    a = best_n_all[obj_i] % len(anchor_mask)

                    obj_mask[b, a, j, i] = 1
                    noobj_mask[b, a, j, i] = 0
                    noobj_mask[b, anchor_ious_masked[obj_i] > self.ignore_thres, i, j] = 0

                    target[b, a, j, i, 0:2] = torch.floor(gt_box[0:2])
                    target[b, a, j, i, 2:4] = torch.log(gt_box[2:4] / masked_anchors[a] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + self.class_indices.index(obj_labels[obj_i].cls)] = 1

                    iou = box_iou(
                        pred[b, a, j, i, :4].unsqueeze(0),
                        box_convert(gt[obj_i].unsqueeze(0), "cxcywh", "xyxy"),
                    )
                    obj_score = pred[b, a, j, i, 4]
                    pred_cls = torch.argmax(pred[b, a, j, i, 5:])
                    t_cls = self.class_indices.index(obj_labels[obj_i].cls)
                    if iou > 0.5 and obj_score > 0.5 and pred_cls == t_cls:
                        n_correct += 1

        if visualize:
            visualize_heatmap(target, pred, output_idx)

        return obj_mask, noobj_mask, target, n_matched_objs, n_correct

    def forward(self, xs, xm, xl, labels = None, visualize = False):
        loss_xy = torch.zeros(3)
        loss_wh = torch.zeros(3)
        loss_conf = torch.zeros(3)
        loss_cls = torch.zeros(3)
        preds = []
        n_labels_total, n_proposals_total, n_correct_total = 0, 0, 0
        for idx, output in enumerate([xs, xm, xl]):
            nb = output.size(0)
            na = len(self.anchor_masks[idx])
            ng = output.size(2)
            nch = 5 + self.nc

            grid_x = self.grid_x[idx].expand(nb, -1, -1, -1)     # Tensor (nb, na, ng, ng)
            grid_y = self.grid_y[idx].expand(nb, -1, -1, -1)     # Tensor (nb, na, ng, ng)
            anchor_w = self.anchor_w[idx].expand(nb, -1, -1, -1) # Tensor (nb, na, ng, ng)
            anchor_h = self.anchor_h[idx].expand(nb, -1, -1, -1) # Tensor (nb, na, ng, ng)

            # [nb, na * (5 + nc), ng, ng] -> [nb, na, ng, ng, (5 + nc)]
            output = output.view(nb, na, nch, ng, ng)
            output = output.permute(0, 1, 3, 4, 2).contiguous()

            # logistic activation for xy, obj, cls
            xy_obj_cls_mask = [0, 1] + [i for i in range(4, nch)]
            output[..., xy_obj_cls_mask] = torch.sigmoid(output[..., xy_obj_cls_mask])

            pred = output.clone()
            pred[..., 0] += grid_x
            pred[..., 1] += grid_y
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h

            if labels is not None:
                n_proposals = (pred[..., 4] > 0.5).sum().item()

                obj_mask, noobj_mask, target, n_matched_objs, n_correct = self.build_target(
                    pred = torch.detach(pred),
                    labels = labels,
                    nch = nch,
                    output_idx = idx,
                    visualize = visualize
                )

                obj_mask = obj_mask.type(torch.ByteTensor).bool()
                noobj_mask = noobj_mask.type(torch.ByteTensor).bool()

                # x and y loss
                loss_xy[idx] = self.lambda_coord * self.mse_loss(input = output[..., :2][obj_mask],
                                                                    target = target[..., :2][obj_mask])

                # width and height loss
                loss_wh[idx] = self.lambda_coord * self.mse_loss(input = output[..., 2:4][obj_mask],
                                                                    target = target[..., 2:4][obj_mask])
                
                # confidence loss
                loss_obj = self.bce_loss(input = output[..., 4][obj_mask],
                                        target = target[..., 4][obj_mask])
                loss_noobj = self.bce_loss(input = output[..., 4][noobj_mask],
                                        target = target[..., 4][noobj_mask])
                loss_conf[idx] = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
                
                # classification loss
                loss_cls[idx] = self.bce_loss(input = output[..., 5:][obj_mask],
                                                    target = target[..., 5:][obj_mask])

                # add num_labels, num_proposals, and num_correct to their respective totals
                n_labels_total += n_matched_objs
                n_proposals_total += n_proposals
                n_correct_total += n_correct
            else:
                pred[..., :4] = pred[..., :4] * self.strides[idx]
                preds.append(pred.view(nb, -1, nch))
        else:
            loss = loss_xy.sum() + loss_wh.sum() + loss_conf.sum() + loss_cls.sum()
            return {
                "total": loss,
                "xy": loss_xy.sum(),
                "wh": loss_wh.sum(),
                "conf": loss_conf.sum(),
                "cls": loss_cls.sum(),
                "n_labels": n_labels_total,
                "n_proposals": n_proposals_total,
                "n_correct": n_correct_total
            }

class MultitaskYOLOLoss(nn.Module):
    def __init__(
        self,
        classes: List[str],
        class_grouping: ClassGrouping,
        anchors: Dict[str, List[List[float]]],
    ) -> None:
        super().__init__()
        self.class_grouping = class_grouping
        self.loss_layers = {
            group_name: YOLOLoss(
                classes,
                class_group,
                group_anchors,
                get_anchor_masks(group_anchors)
            )
            for (group_name, class_group), group_anchors
            in zip(
                class_grouping.groups.items(),
                anchors.values(),
            )
        }
    
    def forward(
        self,
        mt_output: Dict[str, List[Tensor]],
        labels: List[List[ObjectLabel]]
    ) -> Dict[str, Tuple]:
        return {
            group_name: self.loss_layers[group_name](*mt_output[group_name], labels)
            for group_name
            in self.class_grouping.groups.keys()
        }
