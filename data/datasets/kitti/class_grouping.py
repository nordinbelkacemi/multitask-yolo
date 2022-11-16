from data.dataset import ClassGrouping

cgs_all_together = ClassGrouping(
    name="all_together",
    grouping={
        "all": [
            "Car",
            "Cyclist",
            # "DontCare",
            "Misc",
            "Pedestrian",
            "Person_sitting",
            "Tram",
            "Truck",
            "Van",
        ]
    },
    opt_ks=None
)


cgs_logical_sep = ClassGrouping(
    name="logical_sep",
    grouping={
        "vehicle": [
            "Car",
            "Truck",
            "Van",
            "Tram",
        ],
        "person": [
            "Pedestrian",
            "Cyclist",
            "Person_sitting",
        ],
        "other": [
            # "DontCare",
            "Misc",
        ]
    },
    opt_ks={
        "vehicle": 9,
        "person": 8,
        "other": 7,
    }
)

cgs_all_sep = ClassGrouping(
    name="all_sep",
    grouping={
        class_name: [class_name] for class_name in cgs_all_together.grouping["all"]
    },
    opt_ks={
        "Car": 9,
        "Cyclist": 9,
        "Misc": 7,
        "Pedestrian": 5,
        "Person": 8,
        "Tram": 6,
        "Truck": 8,
        "Van": 5,
    }
)