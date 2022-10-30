from util.types import Resolution
import config.config as cfg
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Image as PILImage
from visualization.visualization import get_labeled_img

from data.dataloader import DataLoader
from data.dataset import Dataset
from data.datasets.pascalvoc.metadata import *
from visualization.visualization import unpad_labels

if __name__ == "__main__":
    batch_size = 4
    shuffle = False
    dataset = Dataset.from_name_and_type("pascalvoc", dataset_type="train", shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size)
    

    # ----------------------- Iterating over dataloader -----------------------
    # for i in range(len(dataloader)):
    #     yolo_input = dataloader[i]
    #     print(f"batch {i}\t{yolo_input.image_batch.size()}\t({len(yolo_input.label_batch)})")


    # ---------------------- Visualizing a specific input batch ----------------------
    assert(shuffle == False)

    batch_num = 1
    yolo_input = dataloader[batch_num - 1]
    for id, labels, image in zip(yolo_input.id_batch, yolo_input.label_batch, yolo_input.image_batch):
        # model space
        print(f"id: {id}")
        for label in labels:
            print(label)
        image = transforms.ToPILImage()(image)
        labeled_image = get_labeled_img(image, labels, dataset.classes, scale=2.0)
        labeled_image.save(f"./out/{id}_model_space.jpg")

        # original image space
        original_image = Image.open(f"{dataset.root_path}/{id}.jpg")
        unpadded_labels = unpad_labels(Resolution.from_image(original_image), labels)
        labeled_image_original = get_labeled_img(original_image, unpadded_labels, dataset.classes, scale=2.0)
        labeled_image_original.save(f"./out/{id}_orig_im_space.jpg")
