from config.train_config import train_dataset
from model.loss import MultitaskYOLOLoss
from model.model import get_detections, MultitaskYOLO
from data.dataloader import DataLoader
from util.device import device
import torchvision.transforms as transforms
from PIL import Image
from util.types import Resolution

from visualization.visualization import get_labeled_img, unpad_labels


if __name__ == "__main__":
    dataset = train_dataset
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size)

    m = MultitaskYOLO(dataset.class_grouping, dataset.anchors).to(device)
    loss_fn = MultitaskYOLOLoss(dataset.classes, dataset.class_grouping, dataset.anchors).to(device)

    yolo_input = dataloader[0]

    ids, x, labels = yolo_input.id_batch, yolo_input.image_batch.to(device), yolo_input.label_batch
    y = m(x)
    loss = loss_fn(y, labels)

    for group_name, loss in loss.items():
        print(f"{group_name}:")
        for k, v in loss.items():
            print(f"\t{k} {v}")

    # Prediction and visualization
    # detections = get_detections(y, 0.9, dataset.classes, dataset.class_grouping, dataset.anchors)
    # for id, image, image_detections in zip(ids, x, detections):
    #     original_image = Image.open(f"{dataset.root_path}/{id}.jpg")
    #     image_detections = unpad_labels(Resolution.from_image(original_image), image_detections)
    #     labeled_image = get_labeled_img(original_image, image_detections, dataset.classes, scale=1.5)
    #     labeled_image.save(f"./out/{id}_loss_test.jpg")
