"""
Padding:
    {
        "left": ...,
        "right": ...,
        "top": ...,
        "bottom" ...,
    }
"""

from ctypes import resize
from dataclasses import dataclass
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
from collections import namedtuple
# from tabulate import tabulate
from PIL import Image
import torchvision.transforms.functional as F
# from torchvision import transforms
from torch import Tensor
import numpy as np
import cv2


Resolution = namedtuple("Resolution", ["h", "w"])


def get_padding(resolution: Resolution) -> Dict[str, int]:
    """
    Gets padding needed to make an image square

    Args:
        resolution (Resolution): Resolution of the orginial image (h, w)
    
    Returns:
        Dict[str, int]: The padding described above in the following format:
            {
                "left": ...,
                "right": ...,
                "top": ...,
                "bottom" ...,
            }
    """
    target_w = max(resolution.h, resolution.w)
    target_h = target_w

    padding_left = int((target_w - resolution.w) / 2)
    padding_right = target_w - (resolution.w + padding_left)

    padding_top = int((target_h - resolution.h) / 2)
    padding_bottom = target_h - (resolution.h + padding_top)

    return {
        "left": padding_left,
        "right": padding_right,
        "top": padding_top,
        "bottom": padding_bottom,
    }


def get_labels(file_path: str) -> List[List]:
    """
    Gets labels from an xml file that looks like this:

    <annotation>
        ...
        <object>
            ...
            <bndbox>
                <xmin> ... </xmin>
			    <ymin> ... </ymin>
			    <xmax> ... </xmax>
			    <ymax> ... </ymax>
            </bndbox>
        </object>
        <object>
            ...
        </object>
        ...
    </annotation>

    Args:
        file_path (str): Path to an XML file that contains annotations compliant with the PascalVOC
            format. The relevant parts are described above

    Returns:
        List[List]: A list of object labels, where each object label is the following:
            [class, x1, y1, x2, y2]
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    object_labels = root.findall('object')

    labels_array = []
    for label in object_labels:
        bbox = label.find("bndbox")
        labels_array.append([
            label.find("name").text,
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ])

    return labels_array


def get_padded_img(img: Image, padding: Dict[str, int]) -> Image:
    """
    Pads the input image to make it a square with the resolution given by the target_resolution input variable

    Args:
        file_path (str): The image file's path
        padding (Dict[str, int]): Padding to be applied (in the format described at the top of the module)
    
    Returns:
        Image: The padded image
    """
    return F.pad(
        img=img,
        padding=[
            padding["left"],
            padding["top"],
            padding["right"],
            padding["bottom"]
        ]
    )


def get_padded_labels(labels: List[List], padding: Dict[str, int]) -> List[List]:
    """
    Offsets bounding box coordinates by "left" and "top" attributes of the padding given as input

    Args:
        labels (List[List]): A list of object labels, where each object label is the following:
            [class, x1, y1, x2, y2]
        padding (Dict[str, int]): Padding in the following format:
            {
                "left": ...,
                "right": ...,
                "top": ...,
                "bottom" ...,
            }
        
    Returns:
        labels (List[List]): List of padded object labels in the same format as the input
    """
    padded_labels = [
        [
            label[0],
            label[1] + padding["left"],
            label[2] + padding["top"],
            label[3] + padding["left"],
            label[4] + padding["top"],
        ] for label in labels
    ]
    
    return padded_labels


def get_resized_img(img: Image, target_resolution: Resolution) -> Image:
    """
    Resizes the input image to a target resolution, and returns {"img": Image, "img_tensor": Tensor}
    
    Args:
        img (Image): The image that gets resized
        target_resolution (Resolution): The output resolution

    Returns:
        Image: The resized image
    """
    return img.resize((target_resolution.h, target_resolution.w))


def get_resized_labels(
    labels: List[List],
    input_resolution: Resolution,
    output_resolution: Resolution
) -> List[List]:
    """
    Resizes the labels to the target resolution
    """
    resized_labels = [
        [
            label[0],
            label[1] * output_resolution.w / input_resolution.w,
            label[2] * output_resolution.h / input_resolution.h,
            label[3] * output_resolution.w / input_resolution.w,
            label[4] * output_resolution.h / input_resolution.h
        ] for label in labels
    ]

    return resized_labels


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
        x1, y1, x2, y2 = int(label[1]), int(label[2]), int(label[3]), int(label[4])

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

















img_id = "2007_000032"
label_file_path = f"/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/Annotations/{img_id}.xml"
image_file_path = f"/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/JPEGImages/{img_id}.jpg"
model_input_resolution = Resolution(h=416, w=416)


def get_model_input(
    img: Image,
    labels: List[List],
    target_resolution: Resolution = model_input_resolution
) -> Tuple:
    """
    This method applies square padding to the image and its object labels, then resizes it to the
    given target_resolution (which is the model's input resolution)

    Args:
        img (Image): The input image
        labels (List[List]): A list of object labels, where each object label is the following:
            [class, x1, y1, x2, y2]
        target_resolution (Resolution): The target resolution.
    
    Returns:
        Tuple: A tuple in the format (img, labels), where img is the transformed image (with square
            padding and resizing applied) and labels is the list of transformed object labels.
    """
    image_resolution = Resolution(h=img.size[1], w=img.size[0])
    square_padding = get_padding(image_resolution)
    
    # Apply square padding
    padded_image = get_padded_img(img, square_padding)
    padded_labels = get_padded_labels(labels, square_padding)

    # Resize to model's input resolution
    padded_resolution = Resolution(h=padded_image.size[1], w=padded_image.size[0])
    resized_image = get_resized_img(padded_image, target_resolution)
    resized_labels = get_resized_labels(padded_labels, padded_resolution, target_resolution)

    return resized_image, resized_labels


if __name__ == "__main__":
    raw_image = Image.open(image_file_path)
    raw_labels = get_labels(label_file_path)
    model_input_image, model_input_labels = get_model_input(
        raw_image,
        raw_labels
    )

    labeled_image = get_labeled_img(model_input_image, model_input_labels)
    labeled_image.save(f"./out/{img_id}_loaded.jpg")