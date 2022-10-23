from PIL import Image
from PIL.Image import Image as PILImage
from typing import List
import cv2
import numpy as np

from data.dataset import ObjectLabel
from util.bbox_transforms import scale_bbox, yolo_bbox_to_x1y1x2y2
from util.types import Resolution
import config.config as cfg
from data.transforms import get_padding


def get_labeled_img(original_image: PILImage, labels_model_space: List[ObjectLabel], classes: List[str], visualization_resolution: Resolution) -> PILImage:
    """
    Given object labels in the normalized image space (with the model's input resolution), this
    method draws class labeled bounding boxes around each object on the original image. The output
    is scaled to the desired resolution, which is given by the visualization_resolution parameter.

    Args:
        original_image (Image): Image related to the object labels given by `model_space_labels`
        labels_model_space (List[ObjectLabel]): A list of object labels in yolo format in the
            model's input resolution 
        classes (List[str]): A list of class names
        visualization_resolution (Resolution): Resolution of the output image
    
    Returns:
        Image: The image with ground truth bounding boxes (given by labels) drawn on it in the
            target visualization resolution
    """
    original_resolution = Resolution.from_image(original_image)
    intermediate_resolution = Resolution(
        w=max(original_resolution.w, original_resolution.h),
        h=max(original_resolution.w, original_resolution.h),
    )
    
    padding = get_padding(original_resolution)
    unpadded_labels = [
        ObjectLabel(
            cls=label.cls,
            x=(label.x * intermediate_resolution.w - padding[0]) / original_resolution.w,
            y=(label.y * intermediate_resolution.h - padding[1]) / original_resolution.h,
            w=(label.w * intermediate_resolution.w) / original_resolution.w,
            h=(label.h * intermediate_resolution.h) / original_resolution.h,
        ) for label in labels_model_space
    ]


    image = original_image.resize(visualization_resolution.as_wh_tuple())
    image_cv2 = np.array(image, dtype=np.float32)

    # print(visualization_resolution)
    for label in unpadded_labels:
        cls: str = classes[label.cls]
        [x1, y1, x2, y2] = scale_bbox(
            bbox=yolo_bbox_to_x1y1x2y2(label.get_bbox(), original_resolution),
            scaling_factor_xy=(
                (visualization_resolution.w / original_resolution.w),
                (visualization_resolution.h / original_resolution.h),
            )
        )
        # print(x1, y1, x2, y2)

        # draw bounding box
        cv2.rectangle(
            img=image_cv2,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=(36, 255, 12),
            thickness=1
        )

        # label bounding boxx with the object's class
        cv2.putText(
            img=image_cv2,
            text=cls,
            org=(int(x1), int(y1) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(36, 255, 12),
            thickness=1
        )

    return Image.fromarray(image_cv2.astype(np.uint8))
