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
    'dice': dice_coef,
    'iou': iou_coef,
    'f1_iou': f1_score_at_iou,
}


def get_metric(metric_name: str = 'dice'):
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
            f"Unknown metric: {metric_name}. "
            f"Available: {list(METRICS_FUNCTIONS.keys())}"
        )
    return METRICS_FUNCTIONS[metric_name]
