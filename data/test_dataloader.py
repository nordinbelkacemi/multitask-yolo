from util.types import Resolution
import torchvision.transforms as transforms
from PIL import Image
from visualization.visualization import get_labeled_img

from data.dataloader import DataLoader
from data.datasets.datasets import *
from visualization.visualization import unpad_labels

if __name__ == "__main__":
    dataset = PascalVOCDataset(dataset_type="train")
    dataloader1 = DataLoader(dataset, 4, shuffle=True)
    dataloader2 = DataLoader(dataset, 4, shuffle=True)
    

    # ----------------------- Iterating over dataloader -----------------------
    # for i in range(len(dataloader)):
    #     yolo_input = dataloader[i]
    #     print(f"batch {i}\t{yolo_input.image_batch.size()}\t({len(yolo_input.label_batch)})")


    # ---------------------- Visualizing a specific input batch ----------------------
    batch_num = 1
    # yolo_input = dataloader[batch_num - 1]
    # for id, labels, image in zip(yolo_input.id_batch, yolo_input.label_batch, yolo_input.image_batch):
    #     # model space
    #     print(f"id: {id}")
    #     for label in labels:
    #         print(label)
    #     image = transforms.ToPILImage()(image)
    #     labeled_image = get_labeled_img(image, labels, dataset.classes, scale=2.0)
    #     labeled_image.save(f"./out/{id}_model_space.jpg")

    #     # original image space
    #     original_image = Image.open(f"{dataset.root_path}/{id}.jpg")
    #     unpadded_labels = unpad_labels(Resolution.from_image(original_image), labels)
    #     labeled_image_original = get_labeled_img(original_image, unpadded_labels, dataset.classes, scale=2.0)
    #     labeled_image_original.save(f"./out/{id}_orig_im_space.jpg")

    yolo_input1 = dataloader1[batch_num - 1]
    yolo_input2 = dataloader2[batch_num - 1]

    for id1, labels1, image1 in zip(yolo_input1.id_batch, yolo_input1.label_batch, yolo_input1.image_batch):
        # model space
        print(f"id1: {id1}")

    for id2, labels2, image2 in zip(yolo_input2.id_batch, yolo_input2.label_batch, yolo_input2.image_batch):
        # model space
        print(f"id2: {id2}")
