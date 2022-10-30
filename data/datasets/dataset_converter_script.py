import os
from abc import ABC, abstractmethod
from typing import Dict, List

from PIL import Image
from tqdm import tqdm
from util.bbox_utils import scale_bbox, x1x2y1y2_to_xywh
from util.types import Resolution


def get_image_file_path_from_label_file_path_default(
    label_file_path: str,
    dataset_images_root_path: str,
    image_file_extension: str,
) -> Dict[str, str]:
    """
    Given a label file path, this method returns the associated image's file path. The assumption
    is that the image and label file names are identical (0001.xml is the label of 0001.jpg)
    """
    file_name = label_file_path.split("/")[-1].split(".")[0]
    return f"{dataset_images_root_path}/{file_name}.{image_file_extension}"


class DatasetConverterScript(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_label_paths = self._get_dataset_label_paths()
        self.dataset_image_paths = self._get_dataset_image_paths(self.dataset_label_paths)
        self.classes = self._get_classes()
        _, ext = os.path.splitext(self.dataset_image_paths["train"][0])
        self.image_file_extension = ext[1:]
    
    @abstractmethod
    def _get_dataset_label_paths(self) -> Dict[str, List[str]]:
        """
        Gets the label file paths of the dataset and returns them as a dict:
        {
            "train": [...],
            "val": [...],
        }
        """
        pass

    @abstractmethod
    def _get_dataset_image_paths(self, label_file_paths: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Gets the image file paths of the dataset and returns them as a dict:
        {
            "train": [...],
            "val": [...],
        }
        """
        pass

    @abstractmethod
    def _get_classes(self) -> List[str]:
        """
        Gets the classes of the dataset and returns them as a list of strings:
        [
            "cat",
            "dog",
            ...
        ]
        """
        pass

    @abstractmethod
    def _get_image_file_path_from_label_file_path(self, label_file_path: str) -> str:
        """
        Implements the mapping between a label file path and an image file path
        """
        pass

    @abstractmethod
    def _get_labels(self, label_file_path) -> List[List]:
        """
        Implements the conversion of a dataset object label to the [cls, x1, y1, x2, y2] format

        Args:
            label_file_path (str): The label file's path
        
        Returns:
            List[List]: All object labels in the label file
        """
        pass

    def dataset_labels_to_yolo_labels(
        self,
        dataset_label_path: str
    ) -> str:
        """
        Converts the contents of a pascal voc annotation file to that of a yolo label file.

        Args:
            label_file_path (str): Path to the file that contains the relevant object labels to be
                converted
        
        Returns:
            str: The contents of the yolo label file
        """
        labels = self._get_labels(dataset_label_path)
        out_file_content = ""
        for label in labels:
            # get the image resolution
            image_file_path = self._get_image_file_path_from_label_file_path(
                label_file_path=dataset_label_path
            )
            image_resolution = Resolution.from_image(Image.open(image_file_path))

            # convert class to int
            cls = self.classes.index(label[0])

            # convert bbox to normalized xywh
            x, y, w, h = scale_bbox(
                bbox=x1x2y1y2_to_xywh(bbox=label[1:]),
                scaling_factor_xy=(
                    1 / image_resolution.w,
                    1 / image_resolution.h,
                )
            )

            out_file_content += f"{cls} {x} {y} {w} {h}\n"

        return out_file_content

    def run(
        self,
        dst: str,
        verbose: bool=True,
    ) -> None:
        if os.path.isdir(dst):
            print(f"{dst} already exists. Remove {dst} and try again, or use a different destination path")
            print("\nUsage: python3 -m data.pascalvoc.pascalvoc_to_yolo_converter_script dst\n")
            print("positional arguments:")
            print("dst\t\t\tDestination path; path to the converted dataset")
            return
        
        for dataset_type in ["train", "val"]:
            os.makedirs(f"{dst}/{dataset_type}")

        # convert and create images
        for dataset_type in ["train", "val"]:
            if verbose:
                print(f"Creating all {dataset_type} images at {dst}/{dataset_type}")
            for i, dataset_image_path in enumerate(tqdm(self.dataset_image_paths[dataset_type])):
                Image.open(dataset_image_path).save(f"{dst}/{dataset_type}/{i:06}.jpg")

        # convert and create labels
        for dataset_type in ["train", "val"]:
            if verbose:
                print(f"Creating all {dataset_type} labels at {dst}/{dataset_type}")
            for i, dataset_label_path in enumerate(tqdm(self.dataset_label_paths[dataset_type])):
                with open(f"{dst}/{dataset_type}/{i:06}.txt", "w") as file:
                    yolo_labels_string = self.dataset_labels_to_yolo_labels(dataset_label_path)
                    file.write(yolo_labels_string)
