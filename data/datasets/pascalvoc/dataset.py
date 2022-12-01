from typing import Dict, List
from data.dataset import Dataset
from data.datasets.pascalvoc.class_grouping import *
import config.config as cfg


class PascalVOCDataset(Dataset):
    def __init__(self, dataset_type: str) -> None:
        super().__init__(dataset_type)


    @property
    def name(self) -> str:
        return "pascalvoc"
    

    @property
    def classes(self) -> List[str]:
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]


    @property
    def class_grouping(self) -> ClassGrouping:
        return cgs_all_together
