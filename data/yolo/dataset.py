import torchvision.transforms as transforms
from load_util import get_model_input, get_yolo_labels
from PIL import Image
import torch
from typing import List, Dict
from dataclasses import dataclass
from data.dataset_converter_script import get_image_file_path_from_label_file_path
from glob import glob
from config.config import model_input_resolution
from config.datasets import *


def get_dataset_name(root_path: str) -> List[str]:
    return root_path.split("/")[-1]


def get_label_file_paths(root_path: str) -> List[str]:
        return glob(f"{root_path}/labels/*.txt")


def get_image_file_paths(root_path: str, label_file_paths: List[str]) -> List[str]:
    return [
        get_image_file_path_from_label_file_path(
            label_file_path=label_file_path,
            dataset_images_root_path=f"{root_path}/images",
            image_file_extension=glob(
                f"{root_path}/images/*")[0].split("/")[-1].split(".")[-1]
        ) for label_file_path in label_file_paths
    ]


@dataclass
class DatasetMetadata:
    root_path: str
    classes: List[str]


@dataclass
class ODDatasetItem:
    image: torch.tensor  # (3, h, w)
    labels: torch.tensor  # (n, 5) -> [cls, x1, y1, x2, y2] n times


@dataclass
class YOLODataset():
    root_path: str
    classes: List[str]
    image_file_paths: List[str]
    label_file_paths: List[str]

    def __init__(self, metadata: Dict[str, str]):
        self.root_path = metadata["root_path"]
        self.classes = metadata["classes"]
        self.label_file_paths = get_label_file_paths(metadata["root_path"])
        self.image_file_paths = get_image_file_paths(metadata["root_path"], self.label_file_paths)
 
    def __len__(self) -> int:
        return len(self.label_file_paths)

    def __getitem__(self, index: int) -> ODDatasetItem:
        raw_image = Image.open(self.image_file_paths[index])
        raw_labels = get_yolo_labels(self.label_file_paths[index])

        model_input_image, model_input_labels = get_model_input(raw_image, raw_labels)
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


    



















# pascalvoc dataset


# images_path = "/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/JPEGImages/"
# labels_path = "/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012/Annotations/"
# train_item_ids_file_path = "VOCdevkit/VOC2012/ImageSets/Main/train.txt"
# val_item_ids_file_path = "VOCdevkit/VOC2012/ImageSets/Main/val.txt"


# class PascalVOCDataset(ODDataset):
#     def __init__(self, dataset_type: str):
#         self.classes = [
#             "aeroplane",
#             "bicycle",
#             "bird",
#             "boat",
#             "bottle",
#             "bus",
#             "car",
#             "cat",
#             "chair",
#             "cow",
#             "diningtable",
#             "dog",
#             "horse",
#             "motorbike",
#             "person",
#             "pottedplant",
#             "sheep",
#             "sofa",
#             "train",
#             "tvmonitor",
#         ]

#         item_ids = []

#         if dataset_type == "train":
#             item_ids_file_path = train_item_ids_file_path
#         elif dataset_type == "val":
#             item_ids_file_path = val_item_ids_file_path

#         with open(item_ids_file_path) as file:
#             for line in file:
#                 item_ids.append(line)

#         self.image_file_paths = [
#             f"{images_path}/{item_id}.jpeg" for item_id in item_ids]
#         self.image_file_paths = [
#             f"{labels_path}/{item_id}.xml" for item_id in item_ids]

#     def __len__(self) -> int:
#         len(self.item_ids)

#     def _get_labels(self, index: int) -> List[List]:
#         get_labels(self.label_file_paths[index])
