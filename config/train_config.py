from data.datasets.datasets import *

train_batch_size = 32
eval_batch_size = train_batch_size
train_dataset = PascalVOCDataset("train")
eval_dataset = PascalVOCDataset("val")
num_epochs = 90
lr = 0.001
eval_interval = 3
visualization_interval = 3
