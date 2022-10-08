from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import torch
from PIL import Image
from load_util import get_model_input
import torchvision.transforms as transforms


@dataclass
class ODDatasetItem:
    image: torch.tensor # (3, h, w)
    labels: torch.tensor # (n, 5)


class ODDataset(ABC):
    @property
    @abstractmethod
    def image_file_paths(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def label_file_paths(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def label_file_paths(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def classes() -> List[str]:
        pass
    
    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> ODDatasetItem:
        raw_image = Image.open(self.image_file_paths[index])
        raw_labels = self._get_labels(index)
        
        model_input_image, model_input_labels = get_model_input(
            raw_image,
            raw_labels
        )

        model_input_image_tensor = transforms.ToTensor()(model_input_image)
        model_input_label_tensor = torch.tensor([
            [
                self.classes.index(label[0]),
                label[1],
                label[2],
                label[3],
                label[4],
            ] for label in model_input_labels
        ])

        return ODDatasetItem(
            image=model_input_image_tensor,
            labels=model_input_label_tensor
        )


    @abstractmethod
    def _get_labels(self, index: int) -> List[List]:
        pass
