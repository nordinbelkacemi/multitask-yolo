from abc import ABC, abstractmethod
from random import random
from re import X
from typing import List

import PIL
import torch.nn.functional as F
import torchvision.transforms.functional as F
from PIL import Image
from util.types import Resolution

from data.dataset import DatasetItem, ObjectLabel


class DatasetItemTransform(ABC):
    @abstractmethod
    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        pass


class SquarePadAndResize(DatasetItemTransform):
    """
    Transform class that pads an image to make it square
    """
    def __init__(self, target_resolution: Resolution):
        super.__init__()
        try:
            if target_resolution.w != target_resolution.h:
                raise RuntimeError
            self.target_resolution = target_resolution
        except RuntimeError:
            print("The target resolution must be square")


    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        # Apply padding -> Square image + labels
        padding = self._get_padding(Resolution.from_image(dataset_item.image))
        padded_image = self._get_padded_image(dataset_item.image, padding)
        padded_labels = self._get_padded_labels(dataset_item.labels, padding)

        # Resize image to target resolution
        resized_padded_image = padded_image.resize(
            (self.target_resolution.w, self.target_resolution.h)
        )
        
        return DatasetItem(
            image=resized_padded_image,
            labels=padded_labels  # resize does not change normalized image space object labels
        )


    def _get_padding(self, input_resolution: Resolution) -> List[int]:
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
            padding_right,
            padding_top,
            padding_bottom,
        ]
    
    
    def _get_padded_image(self, image: Image, padding: List[int]) -> Image:
        """
        Applies padding to an input image

        Args:
            image (Image): The input image
            padding (List[int]): Padding for left, top, right and bottom borders respectively
        
        Returns:
            Image: The padded image
        """
        return F.pad(img=image, padding=padding)
    

    def _get_padded_labels(
        self,
        labels: List[ObjectLabel],
        padding: List[int],
        input_image_resolution: Resolution) -> List[ObjectLabel]:
        """
        Offsets bounding box coordinates by the padding for left and top borders

        Args:
            labels (List[ObjectLabel]): Input object labels
            padding (List[int]): Padding for left, top, right and bottom borders respectively
            
        Returns:
            labels (List[ObjectLabel]): Output object labels
        """
        padded_resolution = input_image_resolution.pad(padding)
        padded_labels = [
            ObjectLabel(
                cls=label.cls,
                x=(label.x * input_image_resolution.w + padding[0]) * 1 / padded_resolution.w,
                y=(label.y * input_image_resolution.h + padding[1]) * 1 / padded_resolution.h,
                w=(label.w * input_image_resolution.w) / padded_resolution.w,
                h=(label.h * input_image_resolution.h) / padded_resolution.h,
            ) for label in labels
        ]

        return padded_labels


class RandomHFlip:
    def __init__(self, p: float):
        super.__init__()
        self.p = p
    
    def __call__(self, dataset_item: DatasetItem) -> DatasetItem:
        if random() < self.p:
            return DatasetItem(
                image=dataset_item.image.transpose(PIL.Image.FLIP_LEFT_RIGHT),
                labels=[
                    ObjectLabel(
                        cls=label.cls,
                        x=1 - label.x,
                        y=1 - label.y,
                        w=label.w,
                        h=label.h
                    ) for label in dataset_item.labels
                ],
            )
        else:
            return dataset_item
