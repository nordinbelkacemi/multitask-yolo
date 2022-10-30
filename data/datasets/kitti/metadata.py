original_kitti_root = f"/root/workdir/datasets/kitti"
original_kitti_images_root = f"{original_kitti_root}/training/image_2"
original_kitti_labels_root = f"{original_kitti_root}/training/label_2"


# root path of the dataset in yolo format
kitti_root_path = f"/root/workdir/yolo_datasets/kitti"

kitti_classes = [
    "Car",              # vehicle
    "Truck",            # vehicle
    "DontCare",         # other
    "Pedestrian",       # person
    "Cyclist",          # person
    "Van",              # vehicle
    "Tram",             # vehicle
    "Misc",             # other
    "Person_sitting",   # person
]

kitti_class_groups = {
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

kitti_image_file_extension = "png"