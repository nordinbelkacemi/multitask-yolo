from data.datasets.datasets import *

# Dec_01_09_18_13 dataloader goes through 1 batch   (3 heads)
# Dec_01_09_26_44 dataloader goes through 2 batches (3 heads)
# Dec_01_09_56_54 dataloader goes through 2 batches without bias initialization
# Dec_01_10_09_12 dataloader goes through 2 batches (one head)
# Dec_01_13_24_25 dataloader goes through 2 batches (one head) Adam optimizer not initialized every epoch

# ---- fixed adam reinitialization problem and train loader shuffling ----
# Dec_02_10_06_55 yolov5s losses summed, one head
# Dec_03_14_02_45 yolov5s mean losses, one head, (1 * loss_obj + 2 * loss_noobj) / 3

# ---- KITTI dataset ----

# Dec_03_20_11_09 yolov5s mean losses, one head (all_together), (1 * loss_obj + 2 * loss_noobj) / 3

# ---- added random grayscale and random gaussian blur ----

# Dec_03_23_42_16 yolov5s mean losses, one head (all_together), (1 * loss_obj + 2 * loss_noobj) / 3
# Dec_03_23_50_30 yolov5s mean losses, one head per class (all_sep), (1 * loss_obj + 2 * loss_noobj) / 3 <- IN THESIS
# Dec_04_10_26_08: Dec_03_23_42_16 continued (from ep_53), loss changed to (1 * loss_obj + 5 * loss_noobj) / 6 and nms_iout threshold from 0.5 to 0.6
# Dec_04_11_21_46: Dec_04_10_26_08 continued (changed nms_iou_threshold to 0.2 after the first few epochs)
# Dec_04_23_09_45: yolov5s mean losses, logical sep, (1 * loss_obj + 2 * loss_noobj) / 3 <- HOPEFULY IN THESIS


train_batch_size = 32
overfit = False
eval_batch_size = train_batch_size
train_dataset = KITTIDataset("train")
eval_dataset = KITTIDataset("val") if not overfit else train_dataset
# train_dataset = PascalVOCDataset("train")
# eval_dataset = PascalVOCDataset("val") if not overfit else train_dataset
num_epochs = 300
lr = 0.001
first_eval_epoch = 1
eval_interval = 1
visualization_interval = 20
saved_model_path = None
