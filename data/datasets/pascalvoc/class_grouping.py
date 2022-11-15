cgs_all_together = {
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

cgs_logic_sep = {
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

cgs_all_sep = {
    class_name: [class_name] for class_name in cgs_all_together["all"]
}
