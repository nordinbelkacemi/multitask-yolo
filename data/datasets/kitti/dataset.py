from typing import Dict, List
from data.dataset import Dataset
from data.datasets.kitti.class_grouping import *


class KITTIDataset(Dataset):
    def __init__(self, dataset_type: str, shuffle=True) -> None:
        super().__init__(dataset_type, shuffle)
    

    @property
    def name(self) -> str:
        return "kitti"


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
    def class_grouping(self) -> ClassGrouping:
        return cgs_all_together
