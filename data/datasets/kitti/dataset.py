from typing import Dict, List
from data.dataset import Dataset
from data.datasets.kitti.class_grouping import *


class KITTIDataset(Dataset):
    def __init__(self, dataset_type: str, shuffle=True) -> None:
        super().__init__(dataset_type, shuffle)
    

    @property
    def name(self) -> str:
        "kitti"


    @property
    def root_path(self) -> str:
        return "/root/workdir/yolo_datasets/kitti"


    @property
    def classes(self) -> List[str]:
        return [
            "Car",
            "Cyclist",
            "DontCare",
            "Misc",
            "Pedestrian",
            "Person_sitting",
            "Tram",
            "Truck",
            "Van",
        ]
    

    @property
    def class_groups(self) -> Dict[str, List[str]]:
        return class_groups_1_head
