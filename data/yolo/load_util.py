"""
Padding:
    {
        "left": ...,
        "right": ...,
        "top": ...,
        "bottom" ...,
    }
"""

from typing import Dict, List
from PIL import Image
import torchvision.transforms.functional as F
from config.config import model_input_resolution
from typing import Tuple
from data.dataset_converter_script import get_label_resolution
from util.types import Resolution
import util.bbox_transforms as bbox_transforms

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
    padded_resolution = Resolution(
        h=padded_image.size[1], w=padded_image.size[0])
    resized_image = get_resized_img(padded_image, target_resolution)
    resized_labels = get_resized_labels(
        padded_labels, padded_resolution, target_resolution)

    return resized_image, resized_labels


def get_yolo_labels(label_file_path: str) -> List[List]:
    """
    Gets the labels from the label file of a yolo dataset

    Args:
        label_file_path (str): Path to the label file
    
    Returns:
        List: [cls, x1, y1, x2, y2], where cls is a string
    """
    image_res: Resolution = get_label_resolution(label_file_path)
    labels = []
    with open(label_file_path, "r") as file:
        for line in file:
            obj_label = line.split()

            cls = int(obj_label[0])
            x = float(obj_label[1]) * image_res.w
            y = float(obj_label[2]) * image_res.h
            w = float(obj_label[2]) * image_res.w
            h = float(obj_label[2]) * image_res.h

            bbox_xywh = [x, y, w, h]
            bbox_x1y1x2y2 = bbox_transforms.xywh_to_x1y1x2y2(bbox_xywh)
            label = [cls] + bbox_x1y1x2y2

            labels.append(label)
    
    return labels

        