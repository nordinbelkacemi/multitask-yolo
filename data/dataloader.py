from dataclasses import dataclass
from torch import Tensor
from typing import Optional, List

from transforms import RandomHFlip, SquarePadAndResize
from torchvision.transforms import Compose, ToTensor, Normalize
from dataset import Dataset, ObjectLabel
import random
import torch
import config.config as cfg

@dataclass
class YOLOInput:
    image: Tensor
    """
    Image tensor of shape (b, 3, h, w), where b is the batch size and h and w are the height and
    width of the model's input.
    """
    label: Optional[List[ObjectLabel]]
    """
    Label list of length equal to the batch size, b. This member is `None` in case we run inference
    on the model.
    """


class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset: Dataset = dataset
        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.dataset.item_ids)

        self.transforms = Compose([
            SquarePadAndResize(target_resolution=cfg.model_input_resolution),
            RandomHFlip(p=0.1),
        ])
    
    def __getitem__(self, index: int) -> YOLOInput:
        b, h, w = self.batch_size, cfg.model_input_resolution.h, cfg.model_input_resolution.w

        image_batch = torch.zeros(b, 3, h, w)
        label_batch = [None for _ in range(b)]
        for i in range(self.batch_size * index, self.batch_size * (index + 1)):
            dataset_item = self.transforms(self.dataset[i])
            
            image, labels = dataset_item.image, dataset_item.labels
            image_batch[i] = Compose([
                ToTensor(),
                # Normalize(...)
            ])(image)
            label_batch[i] = labels

        return YOLOInput(image=image_batch, label=label_batch)

