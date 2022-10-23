
from util.types import Resolution
from data.dataloader import DataLoader
from data.datasets.pascalvoc.metadata import *
from data.dataset import Dataset
from PIL import Image
from PIL.Image import Image as PILImage


from visualization.visualization import get_labeled_img


if __name__ == "__main__":
    dataset = Dataset.from_name_and_type("pascalvoc", dataset_type="val", shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4)


    # ----------------------- Iterating over dataloader -----------------------
    # for i in range(len(dataloader)):
    #     yolo_input = dataloader[i]
    #     image_batch, label_batch = yolo_input.image, yolo_input.label
    #     print(f"batch {i}\t{image_batch.size()}\t({len(label_batch)})")


    # ---------------------- Visualizing the first batch ----------------------
    batch_num = 30
    yolo_input = dataloader[batch_num - 1]
    for i, labels in enumerate(yolo_input.label):
        image = Image.open(f"{dataset.root_path}/{(batch_num - 1) * dataloader.batch_size + i:06}.jpg")
        labeled_image: PILImage = get_labeled_img(image, labels, dataset.classes, Resolution.from_image(image))
        labeled_image.save(f"./out/{i:03}.jpg")
        
