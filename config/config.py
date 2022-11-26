from util.types import Resolution


model_input_resolution: Resolution = Resolution(h=640, w=640)
mod_feat_0 = 64

eval_iou_match_threshold = 0.5
detection_score_threshold = 0.5
nms_iou_threshold = 0.5