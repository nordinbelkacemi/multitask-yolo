from PIL import Image
from typing import List
from dataclasses import dataclass


image_file_extension = "jpg"


@dataclass
class ObjectLabel:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class DatasetItem:
    item_id: str
    image: Image
    labels: List[ObjectLabel]


@dataclass
class Dataset:
    root_path: str
    classes: List[str]
    item_ids: List[str]
 
    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> DatasetItem:
        item_id = self. item_ids[index]
        image = Image.open(f"{self.root_path}/{self.item_ids[index]:06}.{image_file_extension}")
        labels = []
        with open(f"{self.root_path}/{self.item_ids[index]:06}.txt", "r") as file:
            for line in file:
                labels.append([
                    int(line[0]),   # cls
                    float(line[1]), # x
                    float(line[2]), # y
                    float(line[3]), # w
                    float(line[4]), # h
                ])
        
        return DatasetItem(item_id=item_id, image=image, labels=labels)
