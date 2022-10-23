from dataclasses import dataclass
from torch import Tensor
from typing import Optional, List, Tuple

from data.transforms import RandomHFlip, SquarePadAndResize
from torchvision.transforms import Compose, ToTensor, Normalize
from data.dataset import Dataset, ObjectLabel
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
    def __init__(self, dataset, batch_size: int):
        self.dataset: Dataset = dataset
        self.batch_size = batch_size

        self.transforms = Compose([
            SquarePadAndResize(target_resolution=cfg.model_input_resolution),
            RandomHFlip(p=0.0),
        ])
    
    def __len__(self):
        return int(len(self.dataset) / self.batch_size)
    
    def __getitem__(self, index: int) -> YOLOInput:
        """
        Gets a batch dataset items
        """
        if not (index < self.__len__()):
            raise IndexError
        
        b, h, w = self.batch_size, cfg.model_input_resolution.h, cfg.model_input_resolution.w

        image_batch = torch.zeros(b, 3, h, w)
        label_batch = [None for _ in range(b)]
        for i in range(self.batch_size * index, self.batch_size * (index + 1)):
            dataset_item = self.transforms(self.dataset[i])
            
            image, labels = dataset_item.image, dataset_item.labels
            image_batch[i % self.batch_size] = Compose([
                ToTensor(),
                # Normalize(...)
            ])(image)
            label_batch[i % self.batch_size] = labels

        return YOLOInput(image=image_batch, label=label_batch)

