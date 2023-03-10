from data.dataset import ClassGrouping


cgs_all_together = ClassGrouping(
    name="all_together",
    groups={
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
    },
    anchor_nums={
        "all": 9
    }
)

cgs_logical_sep = ClassGrouping(
    name="logical_sep",
    groups={
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
    },
    anchor_nums={
        "vehicles": 9,
        "animals": 9,
        "house_objects": 9,
    }
)


cgs_all_sep = ClassGrouping(
    name="all_sep",
    groups={
        class_name: [class_name] for class_name in cgs_all_together.groups["all"]
    },
    anchor_nums={
        "aeroplane": 9,
        "bicycle": 9,
        "bird": 9,
        "boat": 9,
        "bottle": 9,
        "bus": 9,
        "car": 9,
        "cat": 9,
        "chair": 9,
        "cow": 9,
        "diningtable": 9, 
        "dog": 9,
        "horse": 9,
        "motorbike": 9,
        "person": 9,
        "pottedplant": 9,
        "sheep": 9,
        "sofa": 9,
        "train": 9,
        "tvmonitor": 9,
    }
)
