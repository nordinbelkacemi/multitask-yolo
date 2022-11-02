original_pascalvoc_root = f"/root/workdir/datasets/pascalvoc/VOCdevkit/VOC2012"
original_pascalvoc_images_root = f"{original_pascalvoc_root}/JPEGImages"

# root path of the dataset in yolo format
pascalvoc_root_path = "/root/workdir/yolo_datasets/pascalvoc"

pascalvoc_classes = [
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

pascalvoc_image_file_extension = "jpg"
# pascalvoc_train_images_count = 5717
# pascalvoc_val_images_count = 5823

pascalvoc_class_groups = {
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

# experiments:
#   1 head
#   3 heads
#   20 heads (1 for each class)
#   random heads