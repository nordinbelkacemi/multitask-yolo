from data.datasets.datasets import *

train_dataset = PascalVOCDataset(dataset_type="train", shuffle=False)
eval_dataset = PascalVOCDataset("val", shuffle=False)

batch_size = 4