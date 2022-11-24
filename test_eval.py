from eval import eval
from model.model import MultitaskYOLO
from config.train_config import eval_dataset

if __name__ == "__main__":
    model = MultitaskYOLO(eval_dataset.class_grouping, eval_dataset.anchors)
    aps = eval(model, 0, eval_dataset)
    print(aps)