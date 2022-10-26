"""
Label spec:
https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/preparing_data_input.html#label-files

Sample text file:
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Cyclist 0.00 0 - 2.46 665.45 160.00 717.93 217.99 1.72 0.47 1.65 2.45 1.35 22.10 - 2.35
Pedestrian 0.00 2 0.21 423.17 173.67 433.17 224.03 1.60 0.38 0.30 - 5.87 1.63 23.11 - 0.03

Format of each row: cls, _, _, _, xmin, ymin, xmax, ymax, _, _, _, _, _, _, _

code 

images: (training and testing)
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip:
-> data_object_image_2/training/image_2 contains all the image files

labels: (training)
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
-> training/label_2 contains all text files
"""


from typing import List, Dict
from data.datasets.dataset_converter_script import (
    DatasetConverterScript,
    get_image_file_path_from_label_file_path_default
)
import random
from glob import glob
from data.datasets.kitti.metadata import (
    original_kitti_labels_root,
    original_kitti_images_root,
    kitti_root_path,
    kitti_classes,
    kitti_image_file_extension,
)



class KITTIToYOLOConverterScript(DatasetConverterScript):
    def __init__(self) -> None:
        super().__init__()


    def _get_dataset_label_paths(self) -> Dict[str, List[str]]:
        all_label_paths = glob(f"{original_kitti_labels_root}/*.txt")
        num_files = len(all_label_paths)
        random.shuffle(all_label_paths)

        # 80/20 train val split (random)
        return {
            "train": all_label_paths[:int(num_files * 0.8)],
            "val": all_label_paths[int(num_files * 0.8):],
        }
    
    def _get_dataset_image_paths(self, dataset_label_paths: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {
            dataset_type: [
                self._get_image_file_path_from_label_file_path(
                    label_file_path=label_file_path
                ) for label_file_path in dataset_label_paths[dataset_type]
            ] for dataset_type in ["train", "val"]
        }
    
    def _get_classes(self) -> List[str]:
        return kitti_classes

    def _get_image_file_path_from_label_file_path(self, label_file_path: str) -> str:
        return get_image_file_path_from_label_file_path_default(
            label_file_path=label_file_path,
            dataset_images_root_path=original_kitti_images_root,
            image_file_extension=kitti_image_file_extension
        )

    def _get_labels(self, file_path: str) -> List[List]:
        """
        Gets labels from a kitti txt file and returns it in the [[class, x1, y1, x2, y2], ...] format

        Args:
            file_path (str): Path to a kitti txt file

        Returns:
            List[List]: A list of object labels, where each object label is the following:
                [class, x1, y1, x2, y2]
        """
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                class_name, _, _, _, xmin, ymin, xmax, ymax, _, _, _, _, _, _, _ = line.split()
                labels.append([
                    class_name.strip(),
                    float(xmin),
                    float(ymin),
                    float(xmax),
                    float(ymax),
                ])
        return labels


if __name__ == "__main__":
    KITTIToYOLOConverterScript().run(
        dst=kitti_root_path
    )
