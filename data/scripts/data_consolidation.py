import os
from tqdm import tqdm
from data.dataloader import DataLoader
from data.dataset import Dataset
from data.datasets.datasets import *

if __name__ == "__main__":
    pascalvoc_dataset = PascalVOCDataset(dataset_type="train", shuffle=False)
    kitti_dataset = KITTIDataset(dataset_type="train", shuffle=False)
    for dataset in [pascalvoc_dataset, kitti_dataset]:
        dataloader = DataLoader(dataset, batch_size=1)

        if not os.path.exists(f"{dataset.root_path}/anchor_data"):
            os.mkdir(f"{dataset.root_path}/anchor_data")
        else:
            os.remove(f"{dataset.root_path}/anchor_data/{dataset.name}.txt")

        with open(f"{dataset.root_path}/anchor_data/{dataset.name}.txt", "a") as f:
            for i in tqdm(range(len(dataloader))):
                label_batch = dataloader[i].label_batch
                for labels in label_batch:
                    f.write(
                        "".join([f"{label.cls} {label.x} {label.y} {label.w} {label.h}\n" for label in labels])
                    )
