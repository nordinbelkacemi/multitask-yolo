"""
make new dir where the dataset will be stored and cd i
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar


<annotation >
    ...
    <object >
        ...
        <bndbox >
            <xmin > ... < /xmin >
            <ymin > ... < /ymin >
            <xmax > ... < /xmax >
            <ymax > ... < /ymax >
        </bndbox >
    </object >
    <object >
        ...
    </object >
    ...
</annotation>
"""


import xml.etree.ElementTree as ET
from typing import List, Dict
from data.datasets.dataset_converter_script import (
    DatasetConverterScript,
    get_image_file_path_from_label_file_path_default,
)
import sys
from metadata import (
    original_pascalvoc_root,
    original_pascalvoc_images_root,
    pascalvoc_root_path,
    pascalvoc_classes,
    pascalvoc_image_file_extension
)


class PascalVOCToYOLOConverterScript(DatasetConverterScript):
    def __init__(self) -> None:
        super().__init__()


    def _get_dataset_label_paths(self) -> Dict[str, List[str]]:
        return {
            dataset_type: [
                f"{original_pascalvoc_root}/Annotations/{item_id}.xml"
                for item_id in [line.strip() for line in open(f"{original_pascalvoc_root}/ImageSets/Main/{dataset_type}.txt")]
            ] for dataset_type in ["train", "val"]
        }


    def _get_dataset_image_paths(self, dataset_label_paths: Dict[str, List[str]]) -> Dict[str, List[str]]:
        dataset_image_paths = {"train": [], "val": []}

        for dataset_type in ["train", "val"]:
            for dataset_label_path in dataset_label_paths[dataset_type]:
                dataset_image_paths[dataset_type].append(
                    get_image_file_path_from_label_file_path_default(
                        label_file_path=dataset_label_path,
                        dataset_images_root_path=original_pascalvoc_images_root,
                        image_file_extension=pascalvoc_image_file_extension,
                    )
                )

        return dataset_image_paths
    

    def _get_classes(self) -> List[str]:
        return pascalvoc_classes
    

    def _get_image_file_path_from_label_file_path(self, label_file_path: str) -> str:
        return get_image_file_path_from_label_file_path_default(
            label_file_path=label_file_path,
            dataset_images_root_path=original_pascalvoc_images_root,
            image_file_extension=pascalvoc_image_file_extension
        )


    def _get_labels(self, label_file_path: str) -> List[List]:
        """
        Gets labels from an xml file that looks like this:

        <annotation>
            ...
            <object>
                ...
                <bndbox>
                    <xmin> ... </xmin>
                    <ymin> ... </ymin>
                    <xmax> ... </xmax>
                    <ymax> ... </ymax>
                </bndbox>
            </object>
            <object>
                ...
            </object>
            ...
        </annotation>

        Args:
            file_path (str): Path to an XML file that contains annotations compliant with the PascalVOC
                format. The relevant parts are described above

        Returns:
            List[List]: A list of object labels, where each object label is the following:
                [class, x1, y1, x2, y2]
        """
        tree = ET.parse(label_file_path)
        root = tree.getroot()
        object_labels = root.findall('object')

        labels = []
        for label in object_labels:
            bbox = label.find("bndbox")
            labels.append([
                label.find("name").text,
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ])

        return labels
        

if __name__ == "__main__":
    PascalVOCToYOLOConverterScript().run(
        dst=pascalvoc_root_path
    )
