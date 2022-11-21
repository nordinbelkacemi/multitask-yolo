from glob import glob
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Dict, List, Optional
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod
import config.config as cfg
import ast
from config.dataset_locations import yolo_datasets_root_path


image_file_extension = "jpg"


@dataclass
class ClassGrouping:
    name: str
    groups: Dict[str, List[str]]
    anchor_nums: Optional[Dict[str, int]] # can be None before running clustering.              


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
    def bbox(self) -> List[float]:
        """Returns a [x, y, w, h] bbox"""
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
        self.dataset_type = dataset_type
        self.ids = [f"{i:06}" for i in range(len(glob(f"{self.root_path}/*.jpg")))]
        if shuffle:
            random.shuffle(self.ids)


    @property
    def root_path(self) -> str:
        return f"{yolo_datasets_root_path}/{self.name}/{self.dataset_type}"


    @property
    @abstractmethod
    def name(self) -> str:
        pass

    
    @property
    @abstractmethod
    def classes(self) -> List[str]:
        pass


    @property
    @abstractmethod
    def class_grouping(self) -> ClassGrouping:
        pass


    @property
    def anchors(self) -> Dict[str, List[List[float]]]:
        result = {group_name: None for group_name in self.class_grouping.groups.keys()}

        for group_name in self.class_grouping.groups.keys():
            file_name = glob(f"data/datasets/{self.name}/anchors/{self.class_grouping.name}/{group_name}_clustering_*.txt")[0]
            n_a = self.class_grouping.anchor_nums[group_name]
            with open(file_name, "r") as f:
                for line in f:
                    if int(line[0]) == n_a:
                        result[group_name] = ast.literal_eval("".join(line.split(" ")[1:-1]))

        return result

 
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
                        cls=int(line[0]),
                        x=float(line[1]),
                        y=float(line[2]),
                        w=float(line[3]),
                        h=float(line[4]),
                    )
                )
        
        return DatasetItem(id=id, image=image, labels=labels)
