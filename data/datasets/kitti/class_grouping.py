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
        "Misc": 9,
        "Pedestrian": 9,
        "Person_sitting": 9,
        "Tram": 9,
        "Truck": 9,
        "Van": 9,
    }
)