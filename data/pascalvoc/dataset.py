from data.base.dataset import ODDataset
from typing import List
from load_util import get_labels


images_path = "/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/JPEGImages/"
labels_path = "/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/Annotations/"
train_item_ids_file_path = "VOCdevkit/VOC2012/ImageSets/Main/train.txt"
val_item_ids_file_path = "VOCdevkit/VOC2012/ImageSets/Main/val.txt"


class PascalVOCDataset(ODDataset):
    def __init__(self, dataset_type: str):
        self.classes = [
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

        item_ids = []

        if dataset_type == "train":
            item_ids_file_path = train_item_ids_file_path
        elif dataset_type == "val":
            item_ids_file_path = val_item_ids_file_path

        with open(item_ids_file_path) as file:
            for line in file:
                item_ids.append(line)
        
        self.image_file_paths = [
            f"{images_path}/{item_id}.jpeg" for item_id in item_ids]
        self.image_file_paths = [
            f"{labels_path}/{item_id}.xml" for item_id in item_ids]
    
    def __len__(self) -> int:
        len(self.item_ids)

    def _get_labels(self, index: int) -> List[List]:
        get_labels(self.label_file_paths[index])
