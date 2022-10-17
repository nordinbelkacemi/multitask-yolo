from abc import ABC, abstractmethod
from typing import List, Dict
import os
from PIL import Image
from util.types import Resolution
from util.file_util import get_file_name_from_absolute_path
import shutil
from util.bbox_transforms import x1x2y1y2_to_xywh, scale_bbox


# Assumption: Label and image file names have to be identical, eg.
#
#     pascalvoc/.../labels/train/0001.txt, pascalvoc/.../images/0001.jpg
#     the two have the same file name: 0001
#
def get_image_file_path_from_label_file_path_default(
    label_file_path: str,
    dataset_images_root_path: str,
    image_file_extension: str,
) -> Dict[str, str]:
    file_name = label_file_path.split("/")[-1].split(".")[0]
    return f"{dataset_images_root_path}/{file_name}.{image_file_extension}"


# Assumption: Label and image file names have to be identical, eg.
#
#     pascalvoc/.../labels/train/0001.txt, pascalvoc/.../images/0001.jpg
#     the two have the same file name: 0001
#
def get_label_resolution(
    label_file_path: str,
    dataset_images_root_path: str,
    image_file_extension: str,
) -> Resolution:
    """
    Given a label, this method finds the corresponding image and obtains its resolution
    """
    image_file_path = get_image_file_path_from_label_file_path_default(
        label_file_path=label_file_path,
        dataset_images_root_path=dataset_images_root_path,
        image_file_extension=image_file_extension
    )
    image_width, image_height = Image.open(image_file_path).size

    return Resolution(
        w=image_width,
        h=image_height
    )


def create_yolo_dirs(yolo_dataset_root_path: str) -> None:
    """
    Creates the following directory structure:

    - yolo_dataset
        - train
            - images
            - labels
        - val
            - images
            - labels
    
    Args:
        yolo_dataset_root_path (str): Path to the root directory of the new yolo dataset
    """
    for dataset_type in ["train", "val"]:
        os.makedirs(f"{yolo_dataset_root_path}/{dataset_type}/images")
        os.makedirs(f"{yolo_dataset_root_path}/{dataset_type}/labels")

    return (
        f"{yolo_dataset_root_path}/images",
        f"{yolo_dataset_root_path}/labels",
    )


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
        pass

    @abstractmethod
    def _get_image_file_path_from_label_file_path(self, label_file_path: str) -> str:
        pass

    @abstractmethod
    def _get_labels(self, label_file_path) -> List[List]:
        """
        Needs to return [[cls, x1, y1, x2, y2], ...]
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
            w, h = Image.open(
                self._get_image_file_path_from_label_file_path(
                    label_file_path=dataset_label_path
                )
            ).size
            image_resolution = Resolution(w=w, h=h)

            # convert class to int
            cls = self.classes.index(label[0])

            print(label)
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
            print(f"{dst} already exists. Try again with a different path")
            print("\nUsage: python3 -m data.pascalvoc.pascalvoc_to_yolo_converter_script dst\n")
            print("positional arguments:")
            print("dst\t\t\tDestination path; path to the converted dataset")
            return
        
        yolo_dataset_root_path = dst
        create_yolo_dirs(yolo_dataset_root_path)

        # copy images
        for dataset_type in ["train", "val"]:
            for dataset_image_path in self.dataset_image_paths[dataset_type]:
                file_name = get_file_name_from_absolute_path(dataset_image_path)
                shutil.copyfile(
                    src=dataset_image_path,
                    dst=f"{yolo_dataset_root_path}/{dataset_type}/{file_name}.{self.image_file_extension}"
                )
            if verbose:
                print(f"Created all images at {yolo_dataset_root_path}/{dataset_type}")

        # create train and val labels
        for dataset_type in ["train", "val"]:
            for dataset_label_path in self.dataset_label_paths[dataset_type]:
                file_name = get_file_name_from_absolute_path(dataset_label_path)
                with open(f"{yolo_dataset_root_path}/{dataset_type}/{file_name}.txt", "w") as file:
                    yolo_labels_string = self.dataset_labels_to_yolo_labels(dataset_label_path)
                    file.write(yolo_labels_string)
            if verbose:
                print(f"Created all {dataset_type} labels at {yolo_dataset_root_path}/{dataset_type}")