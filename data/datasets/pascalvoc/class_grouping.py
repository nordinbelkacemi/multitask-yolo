class_groups_1_head = {
    "all": [
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
}

class_groups_3_heads = {
    "vehicles": [
        "aeroplane",
        "bicycle",
        "boat",
        "bus",
        "car",
        "motorbike",
        "train",
    ],
    "animals": [
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "person",
        "sheep",
    ],
    "house_objects": [
        "bottle",
        "chair",
        "diningtable",
        "pottedplant",
        "sofa",
        "tvmonitor",
    ]
}

class_groups_20_heads = {
    class_name: [class_name] for class_name in class_groups_1_head["all"]
}
