from data.dataset import ClassGrouping


cgs_all_together = ClassGrouping(
    name="all_together",
    grouping={
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
    opt_ks={
        "all": 6
    }
)

cgs_logical_sep = ClassGrouping(
    name="logical_sep",
    grouping={
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
    opt_ks={
        "vehicles": 6,
        "animals": 8,
        "house_objects": 9,
    }
)


cgs_all_sep = ClassGrouping(
    name="all_sep",
    grouping={
        class_name: [class_name] for class_name in cgs_all_together.grouping["all"]
    },
    opt_ks={
        "aeroplane": 5,
        "bicycle": 9,
        "bird": 6,
        "boat": 9,
        "bottle": 7,
        "bus": 8,
        "car": 7,
        "cat": 8,
        "chair": 6,
        "cow": 9,
        "diningtable": 8, 
        "dog": 7,
        "horse": 9,
        "motorbike": 8,
        "person": 8,
        "pottedplant": 8,
        "sheep": 6,
        "sofa": 8,
        "train": 9,
        "tvmonitor": 8,
    }
)
