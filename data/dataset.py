from glob import glob
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Dict, List
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod


image_file_extension = "jpg"


@dataclass
class ObjectLabel:
    """
    YOLO format object label
    """
    cls: int
    x: float
    y: float
    w: float
    h: float

    @property
    def get_bbox(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class DatasetItem:
    id: str
    labels: List[ObjectLabel]
    image: PILImage


class Dataset(ABC):
    def __init__(self, dataset_type: str, shuffle: bool) -> None:
        """
        Args:
            dataset_type (str): `train` or `val`
            shuffle (bool): Wether items get shuffled or not
        """
        super().__init__()
        self.ids = [f"{i:06}" for i in range(len(glob(f"{self.root_path}/*.jpg")))]
        if shuffle:
            random.shuffle(self.ids)


    @property
    @abstractmethod
    def name(self) -> str:
        pass


    @property
    @abstractmethod
    def root_path(self) -> str:
        pass

    
    @property
    @abstractmethod
    def classes(self) -> List[str]:
        pass


    @property
    @abstractmethod
    def class_groups(self) -> Dict[str, List[str]]:
        pass

 
    def __len__(self) -> int:      
        return len(self.ids)


    def __getitem__(self, index: int) -> DatasetItem:
        if not (index < self.__len__()):
            raise IndexError
        
        id = self.ids[index]
        image = Image.open(f"{self.root_path}/{id}.{image_file_extension}")
        labels = []
        with open(f"{self.root_path}/{id}.txt", "r") as file:
            for line in file:
                line = line.split()
                labels.append(
                    ObjectLabel(
                        cls=int(line[0]),   # cls
                        x=float(line[1]),   # x
                        y=float(line[2]),   # y
                        w=float(line[3]),   # w
                        h=float(line[4]),   # h
                    )
                )
        
        return DatasetItem(id=id, image=image, labels=labels)
