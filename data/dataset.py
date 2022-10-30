from glob import glob
from logging import root
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Dict, List
from dataclasses import dataclass
import random
from data.datasets.kitti.metadata import (
    kitti_root_path,
    kitti_classes,
)
from data.datasets.pascalvoc.metadata import (
    pascalvoc_root_path,
    pascalvoc_classes,
)


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


    def get_bbox(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class DatasetItem:
    id: str
    labels: List[ObjectLabel]
    image: PILImage


class Dataset:
    def __init__(self, root_path: str, classes: List[str], shuffle=True):
        self.root_path = root_path
        self.classes = classes
        self.ids = [f"{i:06}" for i in range(len(glob(f"{root_path}/*.jpg")))]
        if shuffle:
            random.shuffle(self.ids)

 
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
    

    @classmethod
    def from_name_and_type(cls, dataset_name: str, dataset_type: str, shuffle=True):
        """
        Creates a dataset from a dataset name

        Args:
            dataset_name (str): `"pascalvoc"` or `"kitti"`
            dataset_type (str): `"train"` or `"val"`
            shuffle (bool): Wether the dataset items are shuffled on initialization or not
        
        Returns:
            (Dataset) The resulting dataset
        """
        if dataset_name == "pascalvoc":
            return Dataset(
                root_path=f"{pascalvoc_root_path}/{dataset_type}",
                classes=pascalvoc_classes,
                shuffle=shuffle,
            )

        if dataset_name == "kitti":
            return Dataset(
                root_path=f"{kitti_root_path}/{dataset_type}",
                classes=kitti_classes,
                shuffle=shuffle,
            )

