from data.dataset import ClassGrouping

cgs_all_together = ClassGrouping(
    name="all_together",
    groups={
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
    anchor_nums={
        "all": 7
    }
)


cgs_logical_sep = ClassGrouping(
    name="logical_sep",
    groups={
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
    anchor_nums={
        "vehicle": 9,
        "person": 8,
        "other": 7,
    }
)

cgs_all_sep = ClassGrouping(
    name="all_sep",
    groups={
        class_name: [class_name] for class_name in cgs_all_together.groups["all"]
    },
    anchor_nums={
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