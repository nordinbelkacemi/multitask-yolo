from data.datasets.datasets import *

batch_size = 4
train_dataset = PascalVOCDataset("train", shuffle=False)
eval_dataset = train_dataset
# eval_dataset = PascalVOCDataset("val", shuffle=False)
num_epochs = 50
lr = 0.001
eval_interval = 1