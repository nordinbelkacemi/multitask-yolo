original_kitti_root = f"/root/workdir/datasets/kitti"
original_kitti_images_root = f"{original_kitti_root}/training/image_2"
original_kitti_labels_root = f"{original_kitti_root}/training/label_2"


# root path of the dataset in yolo format
kitti_root_path = f"/root/workdir/yolo-datasets/kitti"

kitti_classes = [
    "Car",
    "Truck",
    "DontCare",
    "Pedestrian",
    "Cyclist",
    "Van",
    "Tram",
    "Misc",
    "Person_sitting",
]
kitti_image_file_extension = "png"
