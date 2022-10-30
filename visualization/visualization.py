import PIL
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List
import cv2
import numpy as np

from data.dataset import ObjectLabel
from util.bbox_utils import scale_bbox, yolo_bbox_to_x1y1x2y2
from util.types import Resolution
import config.config as cfg
from data.transforms import get_padding

# unpad_labels(unpadded_resolution, )
# get_labeled_img(image, labels, classes, scale)

def unpad_labels(unpadded_resolution: Resolution, padded_labels: List[ObjectLabel]) -> List[ObjectLabel]:
    """
    Reverts square padded object labels to non padded (ones to be original/unpadded image resolution)
    """
    intermediate_resolution = Resolution(
        w=max(unpadded_resolution.w, unpadded_resolution.h),
        h=max(unpadded_resolution.w, unpadded_resolution.h),
    )
    padding = get_padding(unpadded_resolution)
    return [
        ObjectLabel(
            cls=label.cls,
            x=(label.x * intermediate_resolution.w - padding[0]) / unpadded_resolution.w,
            y=(label.y * intermediate_resolution.h - padding[1]) / unpadded_resolution.h,
            w=(label.w * intermediate_resolution.w) / unpadded_resolution.w,
            h=(label.h * intermediate_resolution.h) / unpadded_resolution.h,
        ) for label in padded_labels
    ]


def get_labeled_img(image: PILImage, labels: List[ObjectLabel], classes: List[str], scale=1.0) -> PILImage:
    """
    Given object labels, this method draws class labeled bounding boxes around each object. The
    output is scaled as per the `scale` parameter.

    Args:
        image (Image): Input image
        labels (List[ObjectLabel]): Object labels on the input image
        classes (List[str]): A list of class names
        scale (float): Resolution of the output image
    
    Returns:
        Image: The image with ground truth bounding boxes (given by labels) drawn on it in the
            target visualization resolution
    """

    w, h = image.size
    image = image.resize((int(w * scale), int(h * scale)))
    image_cv2 = np.array(image, dtype=np.float32)

    # print(visualization_resolution)
    for label in labels:
        cl: str = classes[label.cls]
        [x1, y1, x2, y2] = yolo_bbox_to_x1y1x2y2(
            scale_bbox(label.get_bbox(), (scale, scale)),
            Resolution(w, h)
        )

        # draw bounding box
        cv2.rectangle(
            img=image_cv2,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=(36, 255, 12),
            thickness=2
        )


        # label bounding box with the object's class (green background + black text)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(cl, font_face, font_scale, font_thickness)

        text_x1, text_x2 = int(x1), int(x1 + text_w)
        text_y1, text_y2 = int(y1 - text_h), int(y1)
        if text_y1 < 0:
            text_y1, text_y2 = int(y1), int(y1 + text_h)

        cv2.rectangle(
            img=image_cv2,
            pt1=(text_x1, text_y1),
            pt2=(text_x2, text_y2),
            color=(36, 255, 12),
            thickness=-1,
        )

        cv2.putText(
            img=image_cv2,
            text=cl,
            org=(text_x1, text_y2),
            fontFace=font_face,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=font_thickness,
        )

    return Image.fromarray(image_cv2.astype(np.uint8))
