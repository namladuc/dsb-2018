import numpy as np
import torch
from monai.metrics.utils import get_mask_edges, get_surface_distance

def hausdorff(y_true, y_pred, max_dist):
    result = []
    for i in range(y_true.shape[-1]):
        result.append(1.0 - compute_directed_hausdorff(y_pred[..., i], y_true[..., i], max_dist))
    return np.mean(result)

def hausdorff_slice_first(y_true, y_pred, max_dist, thr=0.5):
    y_true = y_true.to(torch.float32).cpu().detach().numpy()
    y_pred = (y_pred>thr).to(torch.float32).cpu().detach().numpy()
    result = []
    for i in range(y_true.shape[0]):
        result.append(1.0 - compute_directed_hausdorff(y_pred[i, ...], y_true[i, ...], max_dist))
    return np.mean(result)

def compute_directed_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist