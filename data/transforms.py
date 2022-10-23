from abc import ABC, abstractmethod
from random import random
from typing import List

import PIL
import torchvision.transforms.functional as F
from PIL import Image
from PIL.Image import Image as PILImage
from util.types import Resolution

from data.dataset import DatasetItem, ObjectLabel


class DatasetItemTransform(ABC):
    @abstractmethod
    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        pass


def get_padding(input_resolution: Resolution) -> List[int]:
    """
    Gets padding needed to make an image square

    Args:
        resolution (Resolution): Resolution of the orginial image.
    
    Returns:
        List[int]: Padding for left, top, right and bottom borders respectively
    """
    square_w = max(input_resolution.h, input_resolution.w)
    square_h = square_w

    padding_left = int((square_w - input_resolution.w) / 2)
    padding_right = square_w - (input_resolution.w + padding_left)

    padding_top = int((square_h - input_resolution.h) / 2)
    padding_bottom = square_h - (input_resolution.h + padding_top)

    return [
        padding_left,
        padding_top,
        padding_right,
        padding_bottom,
    ]


class SquarePadAndResize(DatasetItemTransform):
    """
    Transform class that pads an image to make it square
    """
    def __init__(self, target_resolution: Resolution):
        try:
            if target_resolution.w != target_resolution.h:
                raise RuntimeError
            self.target_resolution = target_resolution
        except RuntimeError:
            print("The target resolution must be square")


    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        # Apply padding -> Square image + labels
        input_image_resolution = Resolution.from_image(dataset_item.image)
        padding = get_padding(input_image_resolution)
        padded_image = F.pad(img=dataset_item.image, padding=padding)
        padded_resolution = Resolution.from_image(padded_image)
        padded_labels = [
            ObjectLabel(
                cls=label.cls,
                x=(label.x * input_image_resolution.w + padding[0]) / padded_resolution.w,
                y=(label.y * input_image_resolution.h + padding[1]) / padded_resolution.h,
                w=(label.w * input_image_resolution.w) / padded_resolution.w,
                h=(label.h * input_image_resolution.h) / padded_resolution.h,
            ) for label in dataset_item.labels
        ]

        # Resize image to target resolution
        resized_padded_image = padded_image.resize(
            (self.target_resolution.w, self.target_resolution.h)
        )
        
        # return result
        return DatasetItem(
            image=resized_padded_image,
            labels=padded_labels  # resize does not change normalized image space object labels
        )


class RandomHFlip(DatasetItemTransform):
    def __init__(self, p: float):
        self.p = p
    
    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        if random() < self.p:
            return DatasetItem(
                image=dataset_item.image.transpose(PIL.Image.FLIP_LEFT_RIGHT),
                labels=[
                    ObjectLabel(
                        cls=label.cls,
                        x=(1 - label.x),
                        y=label.y,
                        w=label.w,
                        h=label.h
                    ) for label in dataset_item.labels
                ],
            )
        else:
            return dataset_item
