"""Evaluation metrics for 2D segmentation models."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_coef(y_true, y_pred, CFG, thr=0.5, dim=(2, 3), mean_dim=(1, 0), epsilon=0.001):
    """Dice coefficient.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted mask probabilities
        CFG: Configuration object
        thr: Threshold for binarization
        dim: Dimensions to sum over
        mean_dim: Dimensions to average over
        epsilon: Small value for numerical stability

    Returns:
        Dice coefficient (higher is better)
    """
    if CFG.isDeeply:
        y_true = F.interpolate(y_true, size=y_pred.shape[2:])

    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=mean_dim)
    return dice


def iou_coef(y_true, y_pred, CFG, thr=0.5, dim=(2, 3), mean_dim=(1, 0), epsilon=0.001):
    """Intersection over Union (Jaccard Index).

    Args:
        y_true: Ground truth masks
        y_pred: Predicted mask probabilities
        CFG: Configuration object
        thr: Threshold for binarization
        dim: Dimensions to sum over
        mean_dim: Dimensions to average over
        epsilon: Small value for numerical stability

    Returns:
        IoU score (higher is better)
    """
    if CFG.isDeeply:
        y_true = F.interpolate(y_true, size=y_pred.shape[2:])

    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=mean_dim)
    return iou


def f1_score_at_iou(labels_true, labels_pred, iou_threshold=0.7):
    """F1 score based on instance-level IoU matching.

    Uses Hungarian algorithm to find optimal matching between predicted and ground truth instances,
    then computes F1 score based on IoU threshold.

    Args:
        labels_true: Ground truth instance masks [Instances, H, W]
        labels_pred: Predicted instance masks [Instances, H, W]
        iou_threshold: Minimum IoU to consider a match

    Returns:
        F1 score (higher is better)
    """
    if len(labels_true) == 0:
        return 0.0 if len(labels_pred) > 0 else 1.0

    # Compute IoU matrix
    intersection = np.logical_and(labels_true[:, None], labels_pred[None, :]).sum(axis=(2, 3))
    union = np.logical_or(labels_true[:, None], labels_pred[None, :]).sum(axis=(2, 3))
    iou_matrix = intersection / (union + 1e-7)

    # Find optimal matching using Hungarian algorithm
    true_idx, pred_idx = linear_sum_assignment(-iou_matrix)

    # Count true positives
    tp = sum(1 for t, p in zip(true_idx, pred_idx) if iou_matrix[t, p] >= iou_threshold)
    fp = len(labels_pred) - tp
    fn = len(labels_true) - tp

    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    return f1


# Metrics registry
METRICS_FUNCTIONS = {
    "dice": dice_coef,
    "iou": iou_coef,
    "f1_iou": f1_score_at_iou,
}


def dsb2018_map(labels_true, labels_pred):
    """Compute the official DSB-2018 mean Average Precision metric.
    
    Average of Precision at IoU thresholds [0.5, 0.55, ..., 0.95].
    Precision = TP / (TP + FP + FN)
    
    Args:
        labels_true: List of ground truth instance masks [H, W] or labeled image
        labels_pred: List of predicted instance masks [H, W] or labeled image
    """
    if len(labels_true) == 0:
        return 0.0 if len(labels_pred) > 0 else 1.0

    # If input is labeled image, extract instance masks
    if getattr(labels_true, "ndim", 0) == 2:
        from skimage.morphology import label
        labels_true = [labels_true == i for i in range(1, labels_true.max() + 1)]
    if getattr(labels_pred, "ndim", 0) == 2:
        from skimage.morphology import label
        labels_pred = [labels_pred == i for i in range(1, labels_pred.max() + 1)]

    # Convert to arrays for faster sum
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    if len(labels_true) == 0:
        return 0.0 if len(labels_pred) > 0 else 1.0
    if len(labels_pred) == 0:
        return 0.0

    # Convert to float32 for fast matrix multiplication
    true_flat = labels_true.reshape(len(labels_true), -1).astype(np.float32)
    pred_flat = labels_pred.reshape(len(labels_pred), -1).astype(np.float32)
    
    # intersection[i, j] is the count of overlapping pixels between true_i and pred_j
    intersection = np.matmul(true_flat, pred_flat.T)
    
    # union[i, j] = area_i + area_j - intersection[i, j]
    true_areas = true_flat.sum(axis=1)
    pred_areas = pred_flat.sum(axis=1)
    union = true_areas[:, None] + pred_areas[None, :] - intersection
    
    iou_matrix = intersection / (union + 1e-7)

    thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = []

    for t in thresholds:
        # Find matches above threshold
        matches = iou_matrix >= t
        
        # Count TP, FP, FN
        # Each true object and predicted object can match at most once
        tp = 0
        true_matched = np.zeros(len(labels_true), dtype=bool)
        pred_matched = np.zeros(len(labels_pred), dtype=bool)
        
        # Greedy matching (highest IoU first)
        matched_indices = np.argsort(-iou_matrix, axis=None)
        for idx in matched_indices:
            r, c = np.unravel_index(idx, iou_matrix.shape)
            if iou_matrix[r, c] < t: break
            if not true_matched[r] and not pred_matched[c]:
                tp += 1
                true_matched[r] = True
                pred_matched[c] = True
        
        fp = len(labels_pred) - tp
        fn = len(labels_true) - tp
        
        precisions.append(tp / (tp + fp + fn + 1e-7))
        
    return np.mean(precisions)


def dice_numpy(y_true, y_pred, epsilon=1e-7):
    """Dice coefficient for numpy arrays."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection + epsilon) / (y_true.sum() + y_pred.sum() + epsilon)


def get_metric(metric_name: str = "dice"):
    """Get metric function by name.

    Args:
        metric_name: Name of the metric

    Returns:
        Metric function

    Raises:
        ValueError: If metric name is not recognized
    """
    if metric_name not in METRICS_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric_name}. " f"Available: {list(METRICS_FUNCTIONS.keys())}"
        )
    return METRICS_FUNCTIONS[metric_name]
