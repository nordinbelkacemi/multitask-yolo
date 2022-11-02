import matplotlib.pyplot as plt
from data.datasets.pascalvoc.metadata import *
from data.datasets.kitti.metadata import *

if __name__ == "__main__":
    dataset_name = "kitti"
    class_groups = kitti_class_groups
    n_groups = len(class_groups.items())

    fig, axs = plt.subplots(1, n_groups)
    fig.suptitle(dataset_name, fontsize=16)
    fig.set_figwidth(15)
    fig.set_figheight(4)

    x = [3, 4, 5, 6, 7, 8, 9]

    for i, (group_name, classes) in enumerate(class_groups.items()):
        axs[i].title.set_text(group_name)
        if i == 0:
            axs[i].set_ylabel("mean iou")
        axs[i].set_xlabel("k")

        y = []
        with open(f"./out/{dataset_name}_{group_name}_clustering.txt", "r") as f:
            for line in f:
                mean_iou = float(line.split(" ")[-1])
                y.append(mean_iou)
        axs[i].plot(x, y, "-o")
    fig.tight_layout()
    
    fig.savefig(f"./out/{dataset_name}_clustering_results.jpg")
