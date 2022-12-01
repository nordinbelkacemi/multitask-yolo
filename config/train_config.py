from data.datasets.datasets import *

# Dec_01_09_18_13 dataloader goes through 1 batch   (3 heads)
# Dec_01_09_26_44 dataloader goes through 2 batches (3 heads)
# Dec_01_09_56_54 dataloader goes through 2 batches without bias initialization
# Dec_01_10_09_12 dataloader goes through 2 batches (one head)
# Dec_01_13_24_25 dataloader goes through 2 batches (one head) Adam optimizer not initialized every epoch

train_batch_size = 32
overfit = False
eval_batch_size = train_batch_size
train_dataset = PascalVOCDataset("train")
eval_dataset = PascalVOCDataset("val")
num_epochs = 1000
lr = 0.001
first_eval_epoch = 5
eval_interval = 5
visualization_interval = 5
saved_model_path = None
