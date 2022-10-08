from typing import List
import xml.etree.ElementTree as ET

def get_labels(file_path: str) -> List[List]:
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
    tree = ET.parse(file_path)
    root = tree.getroot()
    object_labels = root.findall('object')

    labels_array = []
    for label in object_labels:
        bbox = label.find("bndbox")
        labels_array.append([
            label.find("name").text,
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ])

    return labels_array
