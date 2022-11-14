class_groups_1_head = {
    "all": [
        "Car",
        "Cyclist",
        "DontCare",
        "Misc",
        "Pedestrian",
        "Person_sitting",
        "Tram",
        "Truck",
        "Van",
    ]
}

class_groups_3_heads = {
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
        "DontCare",
        "Misc",
    ]
}

class_groups_9_heads = {
    class_name: [class_name] for class_name in class_groups_1_head["all"]
}