from typing import Dict, List
from model.common import *
import config.config as cfg
import torch.nn as nn
from data.dataset import ClassGrouping, ObjectLabel
from torch import Tensor
from model.loss import MultitaskYOLOLoss
from util.device import device
import config.config as cfg
import math
from torchsummary import summary


class BackBone(nn.Module):
    def __init__(
        self,
        first_ch_out=cfg.mod_feat_0
    ) -> None:
        super().__init__()
        self.d1 = ConvBNSiLU(ch_in=3, ch_out=first_ch_out, k=6, s=2, p=2)                           # 64
        self.d2 = ConvBNSiLU(ch_in=self.d1.ch_out, ch_out=self.d1.ch_out * 2, k=3, s=2, p=1)        # 128
        self.c3_1 = C3(ch_in=self.d2.ch_out, ch_out=self.d2.ch_out, n=1, shortcut=True, e=0.5)      # 128
        self.d3 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out * 2, k=3, s=2, p=1)    # 256
        self.c3_2 = C3(ch_in=self.d3.ch_out, ch_out=self.d3.ch_out, n=2, shortcut=True, e=0.5)      # 256
        self.d4 = ConvBNSiLU(ch_in=self.c3_2.ch_out, ch_out=self.c3_2.ch_out * 2, k=3, s=2, p=1)    # 512
        self.c3_3 = C3(ch_in=self.d4.ch_out, ch_out=self.d4.ch_out, n=3, shortcut=True, e=0.5)      # 512
        self.d5 = ConvBNSiLU(ch_in=self.c3_3.ch_out, ch_out=self.c3_3.ch_out * 2, k=3, s=2, p=1)    # 1024
        self.c3_4 = C3(ch_in=self.d5.ch_out, ch_out=self.d5.ch_out, n=1, shortcut=True, e=0.5)      # 1024
    
    def forward(self, x):
        x1 = self.c3_2(self.d3(self.c3_1(self.d2(self.d1(x)))))
        x2 = self.c3_3(self.d4(x1))
        x3 = self.c3_4(self.d5(x2))
        return [x1, x2, x3]


class Neck(nn.Module):
    def __init__(
        self,
        ch_in=cfg.mod_feat_0 * 16,
        first_ch_out=cfg.mod_feat_0 * 16,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.sppf = SPPF(ch_in=ch_in, ch_out=first_ch_out)
        self.cv1 = ConvBNSiLU(ch_in=self.sppf.ch_out, ch_out=self.sppf.ch_out // 2, k=1, s=1, p=0)
        self.c3_1 = C3(ch_in=self.cv1.ch_out * 2, ch_out=self.cv1.ch_out, n=1, shortcut=False, e=1.0)
        self.cv2 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out // 2, k=1, s=1, p=0)
        self.c3_2 = C3(ch_in=self.cv2.ch_out * 2, ch_out=self.cv2.ch_out, n=1, shortcut=False, e=1.0)
        
    def forward(self, x1, x2, x3):
        x4 = self.cv1(self.sppf(x3))
        x5 = self.cv2(self.c3_1(torch.cat([x2, self.up(x4)], dim=1)))
        x6 = self.c3_2(torch.cat([x1, self.up(x5)], dim=1))
        return [x6, x5, x4]


class Head(nn.Module):
    def __init__(
        self,
        ch_in=cfg.mod_feat_0 * 4,
        first_ch_out=cfg.mod_feat_0 * 4
    ) -> None:
        super().__init__()
        self.cv1 = ConvBNSiLU(ch_in=ch_in, ch_out=first_ch_out, k=3, s=2, p=1)
        self.c3_1 = C3(ch_in=self.cv1.ch_out * 2, ch_out=self.cv1.ch_out * 2, n=1, shortcut=False, e=0.5)
        self.cv2 = ConvBNSiLU(ch_in=self.c3_1.ch_out, ch_out=self.c3_1.ch_out, k=3, s=2, p=1)
        self.c3_2 = C3(ch_in=self.cv2.ch_out * 2, ch_out=self.cv2.ch_out * 2, n=1, shortcut=False, e=0.5)
    
    def forward(self, x6, x5, x4):
        xs = x6
        xm = self.c3_1(torch.cat([x5, self.cv1(xs)], dim=1))
        xl = self.c3_2(torch.cat([x4, self.cv2(xm)], dim=1))
        return [xs, xm, xl]
    
class YOLOv5(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, x):
        [xs, xm, xl] = self.head(*self.neck(*self.backbone(x)))
        return [xs, xm, xl]


class MultitaskYOLO(nn.Module):
    def __init__(
        self,
        class_grouping: ClassGrouping,
        anchors: Dict[str, List[List[float]]]
    ) -> None:
        super().__init__()
        self.yolov5 = YOLOv5()
        summary(self.yolov5.to(device), (3, 416, 416))
        self.mt_heads = nn.ModuleDict({
            group_name: nn.ModuleList([
                nn.Conv2d(
                    cfg.mod_feat_0 * 4,
                    (5 + len(classes)) * len(get_anchor_masks(group_anchors)[0]),
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Conv2d(
                    cfg.mod_feat_0 * 8,
                    (5 + len(classes)) * len(get_anchor_masks(group_anchors)[1]),
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Conv2d(
                    cfg.mod_feat_0 * 16,
                    (5 + len(classes)) * len(get_anchor_masks(group_anchors)[2]),
                    kernel_size=1, stride=1, padding=0
                ),
            ])
            for (group_name, classes), group_anchors
            in zip(class_grouping.groups.items(), anchors.values())
        })

        # initialize biases to output 0.01 score after sigmoid
        p = 0.01
        for conv_ms, group_anchors in zip(self.mt_heads.values(), anchors.values()):
            for conv_m, a_mask in zip(conv_ms, get_anchor_masks(group_anchors)):
                na = len(a_mask)
                b = conv_m.bias.view(na, -1)
                b.data[:, 4] = -math.log((1 - p) / p)
                conv_m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    
    def forward(self, x) -> Dict[str, List[Tensor]]:
        [xs, xm, xl] = self.yolov5(x)
        return {
            group_name: [
                mt_head[i](x_i)
                for i, x_i
                in enumerate([xs, xm, xl])
            ]
            for group_name, mt_head
            in self.mt_heads.items()
        }     


def get_detections(
    preds: Dict[str, Tensor],
    score_thres: float,
    all_classes: List[str],
    class_grouping: ClassGrouping,
) -> List[List[ObjectLabel]]:
    """Merges predictions coming from several multitask heads into a single tensor that contains
    all predictions whose confidence score is above conf_thes
    
    Args:
        preds (Dict[str, Tensor]): {"gp_1": preds_1, ..., "gp_n": preds_n} (all predictions for
            each group)
        score_thres (float): If a prediction's score is below the score threshol, it is omitted
        all_classes (List[str]): all classes
        class_grouping (ClassGrouping): call grouping

    Returns:
        List[List[ObjectLabel]]: nb lists of n ObjectLabels, where bn is the batch size, n is the
        number of predictions above the confidence threshold. (ObjectLabels are in YOLO format).
    """
    nb = list(preds.values())[0].size(0)

    for group_name, class_group in class_grouping.groups.items():
        boxes_and_scores = preds[group_name][:, :, :5]
        class_indices = [all_classes.index(class_name) for class_name in class_group]
        n = preds[group_name].size(1)
        class_vectors = torch.zeros(nb, n, len(all_classes)).to(device)
        class_vectors[:, :, class_indices] = preds[group_name][:, :, 5:]

        preds[group_name] = torch.cat([boxes_and_scores, class_vectors], dim=-1)
    
    detections: Tensor = torch.cat([
        group_preds
        for group_preds
        in preds.values()
    ], dim=1)

    
    return [
        [
            ObjectLabel(
                cls=det[5:].argmax().item(),
                x=det[0].item() / cfg.model_input_resolution.w,
                y=det[1].item() / cfg.model_input_resolution.w,
                w=det[2].item() / cfg.model_input_resolution.w,
                h=det[3].item() / cfg.model_input_resolution.w,
            )
            for det in detections[b][detections[b][:, 4] >= score_thres]
        ]
        for b in range(nb)
    ]
