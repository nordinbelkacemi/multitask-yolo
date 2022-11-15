from typing import Dict, List
from data.dataset import Dataset
from data.datasets.pascalvoc.class_grouping import *


class PascalVOCDataset(Dataset):
    def __init__(self, dataset_type: str, shuffle=True) -> None:
        super().__init__(dataset_type, shuffle)


    @property
    def name(self) -> str:
        return "pascalvoc"


    @property
    def root_path(self) -> str:
        return "/root/workdir/yolo_datasets/pascalvoc"
    

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
    def class_groups(self) -> Dict[str, List[str]]:
        return cgs_logic_sep
