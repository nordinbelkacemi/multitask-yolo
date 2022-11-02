from typing import List, Tuple
from data.dataset import Dataset
import numpy as np
from matplotlib import pyplot as plt
from data.datasets.pascalvoc.metadata import *
from data.datasets.kitti.metadata import *
from torchvision.ops.boxes import box_iou
from torch import Tensor
import torch
from tqdm import tqdm


def bboxes_to_origin(bboxes: Tensor) -> Tensor:
    """
    [x, y, w, h] -> [0, 0, w, h]

    Args:
        bboxes (Tensor): Tensor of shape (n, 5) containing [cls, x, y, w, h] boxes
    
    Returns:
        Tensor: Bboxes with the same w and h as the input bboxes, but with x, y = 0
    """
    return torch.cat([torch.zeros(len(bboxes), 2), bboxes[:, 2:]], dim=1)


def filter_bboxes(bboxes: Tensor, class_indices: List[int]) -> Tensor:
    """
    Selects bboxes whose class is in the `class_indices` list:

    Args:
        bboxes (Tensor): Tensor of shape (n, 5) containing [cls, x, y, w, h] boxes
        class_indices (List[int]): Class indices that we wish to select
    
    Returns:
        Tensor of bboxes whose class is in the `class_indices` list.
    """
    masks = torch.tensor([(bboxes[:,0] == i).tolist() for i in class_indices])
    mask = torch.any(masks, dim=0)
    return bboxes[mask]


def get_clustering(bboxes: Tensor, centroid_boxes: Tensor) -> Tensor:
    return torch.tensor([
        torch.argmax(box_iou(
            bbox.unsqueeze(0),
            centroid_boxes
        ).squeeze()) for bbox in bboxes
    ])


def kmeans_iou_dist(bboxes: Tensor, k: int, iters=3, verbose=True) -> Tuple[Tensor, float]:
    """
    Performs k means clustering for `iters` iterations with k clusters.

    Args:
        bboxes (Tensor): `[[0, 0, w, h], ..., [0, 0, w, h]]`
        k (int): number of cluster centroids
        iters: number of iterations to run
    
    Returns:
        Tuple[Tensor, float]: Cluster centroids `[[w, h], ..., [w, h]]` (k, 2) and mean iou (float)
    """
    
    n = len(bboxes)
    best_centroids_wh, best_mean_iou = None, 0
    
    for _ in range(iters):
        centroid_boxes = bboxes[torch.randperm(n)[:k]]
        prev_clustering = get_clustering(bboxes, centroid_boxes)
        prev_masks = [prev_clustering == i for i in range(k)]
        while True:
            centroid_boxes = torch.tensor([torch.mean(bboxes[mask], dim=0).tolist()
                                           for mask
                                           in prev_masks])
            new_clustering = get_clustering(bboxes, centroid_boxes)
            new_masks = [new_clustering == i for i in range(k)]
            n_changed = torch.count_nonzero(new_clustering != prev_clustering)
            if verbose:
                print([f"{torch.count_nonzero(new_clustering == i)}" for i in range(k)])
                print(f"{n_changed} points changed cluster groups ({(n_changed / n * 100):2f}%)\n")

            if n_changed < int(n * 0.005):
                # centroids
                centroids_wh = centroid_boxes[:, 2:]

                # mean iou
                ious = torch.tensor([box_iou(bbox.unsqueeze(0), centroid_boxes[cluster_idx].unsqueeze(0)).item()
                                     for bbox, cluster_idx
                                     in zip(bboxes, new_clustering)])
                mean_iou = torch.mean(ious)

                if mean_iou > best_mean_iou: 
                    best_centroids_wh = centroids_wh
                    best_mean_iou = mean_iou

                break
            else:
                prev_clustering = new_clustering
                prev_masks = new_masks
    
    return best_centroids_wh, best_mean_iou


def run_group_clustering(bboxes: Tensor, ks: List[int], group_name: str, class_group_i: List[int], dataset_name: str) -> None:
    print(group_name)
    group_bboxes = filter_bboxes(bboxes, class_group_i)
    group_bboxes_origin = bboxes_to_origin(group_bboxes[:, 1:])

    fig, ax = plt.subplots()
    y = []
    with open(f"./out/{dataset_name}_{group_name}_clustering.txt", "a") as f:
        for k in ks:
            print(f"k={k}")
            normalized_anchors, mean_iou = kmeans_iou_dist(group_bboxes_origin, k, verbose=True)
            y.append(mean_iou)
            f.write(f"{k} {normalized_anchors.tolist()} {mean_iou}\n")
            print(f"anchors={normalized_anchors.tolist()}, mean_iou={mean_iou}")
    ax.plot(ks, y, "-o")
    fig.savefig(f"./out/{dataset_name}_{group_name}_clustering.jpg")


if __name__ == "__main__":
    # dataset_name = "kitti"
    # class_groups = kitti_class_groups

    dataset_name = "pascalvoc"
    class_groups = pascalvoc_class_groups

    dataset = Dataset.from_name_and_type(dataset_name, dataset_type="train", shuffle=False)
    classes = dataset.classes
    bboxes = torch.from_numpy(np.loadtxt(f"{dataset.root_path}/anchor_data/{dataset_name}.txt"))
    class_groups_i = {
        group_name: [classes.index(name) for name in class_names]
        for group_name, class_names in class_groups.items()
    }

    # for idx, (group_name, class_group_i) in enumerate(class_groups_i.items()):
    #     run_group_clustering(
    #         bboxes=bboxes,
    #         ks=[3, 4, 5, 6, 7, 8, 9],
    #         group_name=group_name,
    #         class_group_i=class_group_i,
    #         dataset_name=dataset_name,
    #     )


    run_group_clustering(
        bboxes=bboxes,
        ks=[3, 4, 5, 6, 7, 8, 9],
        group_name="house_objects",
        class_group_i=[dataset.classes.index(name) for name in pascalvoc_class_groups["house_objects"]],
        dataset_name=dataset_name,
    )
