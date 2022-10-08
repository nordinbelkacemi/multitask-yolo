from PIL import Image
from typing import List
import cv2
import numpy as np

def get_labeled_img(img: Image, labels: List[List]) -> Image:
    """
    Draws bounding boxes given by labels onto the input image and returns the result

    Args:
        img (Image): The input image
        labels (List[List]): A list of object labels, where each object label is the following:
            [class, x1, y1, x2, y2]
    
    Returns:
        Image: The image with ground truth bounding boxes (given by labels) drawn on it.
    """
    img_cv2 = cv2.cvtColor(np.array(img, dtype=np.float32), cv2.COLOR_RGB2BGR)

    for label in labels:
        cls: str = label[0]
        x1, y1, x2, y2 = int(label[1]), int(
            label[2]), int(label[3]), int(label[4])

        # draw bounding box
        cv2.rectangle(
            img=img_cv2,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(36, 255, 12),
            thickness=1
        )

        # label bounding boxx with the object's class
        cv2.putText(
            img=img_cv2,
            text=cls,
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.2,
            color=(36, 255, 12),
            thickness=1
        )

    return Image.fromarray(img_cv2.astype(np.uint8))
