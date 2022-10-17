from typing import List, Tuple


def x1x2y1y2_to_xywh(bbox: List) -> List:
    """
    Converts (x1, y1, x2, y2) bbox to (x, y, w, h)

    Args:
        bbox (List): [x1, y1, x2, y2]
    
    Returns:
        List: [x, y, w, h]
    """
    return [
        (bbox[0] + bbox[2]) / 2,        # x = (x1 + x2) / 2
        (bbox[1] + bbox[3]) / 2,        # y = (y1 + y2) / 2
        bbox[2] - bbox[0],              # w = x2 - x1
        bbox[3] - bbox[1],              # h = y2 - y1
    ]


def xywh_to_x1y1x2y2(bbox: List) -> List:
    """
    Converts (x, y, w, h) bbox to (x1, y1, x2, y2)

    Args:
        bbox (List): [x, y, w, h]
    
    Returns:
        List: [x1, y1, x2, y2]
    """
    return [
        bbox[0] - bbox[2] / 2,          # x1 = x - w / 2
        bbox[1] - bbox[3] / 2,          # y1 = y - h / 2
        bbox[0] + bbox[2] / 2,          # x2 = x + w / 2
        bbox[1] + bbox[3] / 2,          # y2 = y + h / 2
    ]


def scale_bbox(bbox: List, scaling_factor_xy: Tuple[float, float]) -> List:
    """
    Args:
        bbox (List): [x1, y1, x2, y2]
        scaling_factor_xy (Tuple[float, float]): (scaling_factor_x, scaling_factor_y)
    
    Returns:
        List: [x, y, w h]
    """
    return [
        bbox[0] * scaling_factor_xy[0],  # x * scaling_factor_x
        bbox[1] * scaling_factor_xy[1],  # y * scaling_factor_y
        bbox[2] * scaling_factor_xy[0],  # w * scaling_factor_x
        bbox[3] * scaling_factor_xy[1],  # h * scaling_factor_y
    ]
