from typing import List, Tuple
from util.types import Resolution
import torch
import math

# source: https://github.com/ultralytics/yolov5/blob/2026d4c5eb4e3e48b5295106db85c844000d95d1/utils/general.py#L188-L231
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def x1x2y1y2_to_xywh(bbox: List[float]) -> List[float]:
    """
    Converts `[x1, y1, x2, y2]` bbox to `[x, y, w, h]`.

    Args:
        bbox (List[float]): `[x1, y1, x2, y2]` bbox
    
    Returns:
        List[float]: [x, y, w, h] bbox
    """
    return [
        (bbox[0] + bbox[2]) / 2,        # x = (x1 + x2) / 2
        (bbox[1] + bbox[3]) / 2,        # y = (y1 + y2) / 2
        bbox[2] - bbox[0],              # w = x2 - x1
        bbox[3] - bbox[1],              # h = y2 - y1
    ]


def xywh_to_x1y1x2y2(bbox: List[float]) -> List[float]:
    """
    Converts `[x, y, w, h]` bbox to `[x1, y1, x2, y2]`

    Args:
        bbox (List[float]): `[x, y, w, h]` bbox
    
    Returns:
        List[float]: `[x1, y1, x2, y2]` bbox
    """
    return [
        bbox[0] - bbox[2] / 2,          # x1 = x - w / 2
        bbox[1] - bbox[3] / 2,          # y1 = y - h / 2
        bbox[0] + bbox[2] / 2,          # x2 = x + w / 2
        bbox[1] + bbox[3] / 2,          # y2 = y + h / 2
    ]


def scale_bbox(bbox: List[float], scaling_factor_xy: Tuple[float, float]) -> List[float]:
    """
    Scales an `[x, y, w, h]` bbox by an x and y scaling factor (both in one parameter; `scaling_factor_xy`).

    Args:
        bbox (List[float]): `[x, y, w, h]`
        scaling_factor_xy (Tuple[float, float]): `(scaling_factor_x, scaling_factor_y)`
    
    Returns:
        List[float]: `[x, y, w, h]`
    """
    return [
        bbox[0] * scaling_factor_xy[0],  # x * scaling_factor_x
        bbox[1] * scaling_factor_xy[1],  # y * scaling_factor_y
        bbox[2] * scaling_factor_xy[0],  # w * scaling_factor_x
        bbox[3] * scaling_factor_xy[1],  # h * scaling_factor_y
    ]


def yolo_bbox_to_x1y1x2y2(bbox: List[float], image_resolution: Resolution) -> List[float]:
    """
    Converts a YOLO bbox (normalized image space `[x, y, w, h]` bbox) to an image space
    `[x1, y1, x2, y2]` bbox.

    Args:
        bbox (List[float]): YOLO `[x, y, w, h]` bbox
    
    Returns:
        List[float]: image space `[x1, x2, y1, y2]`
    """
    return xywh_to_x1y1x2y2(scale_bbox(bbox, image_resolution.as_wh_tuple()))