from dataclasses import dataclass
from torch import Tensor
from typing import Optional, List

from data.transforms import RandomHFlip, SquarePadAndResize
from torchvision.transforms import Compose, ToTensor, Normalize
from data.dataset import Dataset, ObjectLabel
import torch
import config.config as cfg
from config.train_config import overfit

@dataclass
class YOLOInput:
    id_batch: List[str]
    """
    Contains the dataset item id of each input batch item.
    """
    label_batch: Optional[List[ObjectLabel]]
    """
    Label list of length equal to the batch size, b. This member is `None` in case we run inference
    on the model.
    """
    image_batch: Tensor
    """
    Image tensor of shape (b, 3, h, w), where b is the batch size and h and w are the height and
    width of the model's input.
    """


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, p_hflip=0.0):
        self.dataset: Dataset = dataset
        self.batch_size = batch_size

        n = len(dataset)
        self.selection_indices = torch.arange(n).tolist()
        if shuffle:
            self.selection_indices = torch.randperm(n).tolist()

        self.space_transforms = Compose([
            SquarePadAndResize(target_resolution=cfg.model_input_resolution),
            RandomHFlip(p=p_hflip),
        ])
        self.final_image_transforms = Compose([
            ToTensor(),
            # Normalize(...),
        ])
    

    def __len__(self):
        if overfit:
            return 16
        else:
            return int(len(self.dataset) / self.batch_size)
    
    
    def __getitem__(self, index: int) -> YOLOInput:
        """
        Gets a batch dataset items
        """
        if index >= self.__len__():
            raise IndexError
        
        b, h, w = self.batch_size, cfg.model_input_resolution.h, cfg.model_input_resolution.w

        id_batch = [None for _ in range(b)]
        label_batch = [None for _ in range(b)]
        image_batch = torch.zeros(b, 3, h, w)
        for i in range(self.batch_size * index, self.batch_size * (index + 1)):
            idx = self.selection_indices[i]
            dataset_item = self.space_transforms(self.dataset[idx])
            
            id_batch[i % self.batch_size] = dataset_item.id
            label_batch[i % self.batch_size] = dataset_item.labels
            image_batch[i % self.batch_size] = self.final_image_transforms(dataset_item.image)

        return YOLOInput(id_batch, label_batch, image_batch)

