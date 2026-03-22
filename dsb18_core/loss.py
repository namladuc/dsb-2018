"""Loss functions for 2D segmentation models."""

import torch.nn.functional as F
import segmentation_models_pytorch as smp


def _align_binary_shapes(y_pred, y_true, CFG):
    """Align prediction and target tensors to (B, C, H, W) for binary losses."""
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if CFG.isDeeply and y_true.shape[2:] != y_pred.shape[2:]:
        y_true = F.interpolate(y_true, size=y_pred.shape[2:])

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Prediction and target must have the same shape after alignment, "
            f"got y_pred={tuple(y_pred.shape)} and y_true={tuple(y_true.shape)}"
        )

    return y_pred, y_true


def criterion_dice_entropy(y_pred, y_true, CFG):
    """Dice + Entropy loss (50% BCE + 50% Dice)."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return 0.5 * smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true) + 0.5 * smp.losses.DiceLoss(
        mode="binary"
    )(y_pred, y_true)


def criterion_tversky_entropy(y_pred, y_true, CFG):
    """Tversky + Entropy loss (50% BCE + 50% Tversky)."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return 0.5 * smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true) + 0.5 * smp.losses.TverskyLoss(
        mode="binary", log_loss=False
    )(y_pred, y_true)


def criterion_focal_dice(y_pred, y_true, CFG):
    """Focal + Dice loss (50% Focal + 50% Dice)."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return 0.5 * smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)(
        y_pred, y_true
    ) + 0.5 * smp.losses.DiceLoss(mode="binary")(y_pred, y_true)


def criterion_bce(y_pred, y_true, CFG):
    """Binary Cross Entropy loss."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true)


def criterion_dice(y_pred, y_true, CFG):
    """Dice loss only."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return smp.losses.DiceLoss(mode="binary")(y_pred, y_true)


def criterion_tversky(y_pred, y_true, CFG):
    """Tversky loss only."""
    y_pred, y_true = _align_binary_shapes(y_pred, y_true, CFG)
    return smp.losses.TverskyLoss(mode="binary", log_loss=False)(y_pred, y_true)


# Loss function registry
LOSS_FUNCTIONS = {
    "dice_entropy": criterion_dice_entropy,
    "tversky_entropy": criterion_tversky_entropy,
    "focal_dice": criterion_focal_dice,
    "bce": criterion_bce,
    "dice": criterion_dice,
    "tversky": criterion_tversky,
}


def get_criterion(loss_name: str = "dice_entropy"):
    """Get loss function by name.

    Args:
        loss_name: Name of the loss function

    Returns:
        Loss function

    Raises:
        ValueError: If loss function name is not recognized
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function: {loss_name}. " f"Available: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[loss_name]


def criterion(y_pred, y_true, CFG):
    """Default criterion wrapper using CFG.loss_name."""
    loss_name = getattr(CFG, "loss_name", "dice_entropy")
    loss_fn = get_criterion(loss_name)
    return loss_fn(y_pred, y_true, CFG)
