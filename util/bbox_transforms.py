from typing import List, Tuple
from util.types import Resolution


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