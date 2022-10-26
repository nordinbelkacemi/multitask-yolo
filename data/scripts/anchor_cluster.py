from data.dataset import Dataset
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dataset = Dataset.from_name_and_type("pascalvoc", dataset_type="train", shuffle=False)
    model_space_labels_all = np.loadtxt(f"{dataset.root_path}/anchor_data/msl_consolidated.txt")
    
    fig, ax = plt.subplots()
    ax.scatter(w, h, vmin=0, vmax=1)
    plt.savefig("out/pascalvoc_all_wh_plot.jpg")
