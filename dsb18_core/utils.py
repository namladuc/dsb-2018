import os
import torch
import random
import numpy as np
from torch.optim import lr_scheduler


def set_seed(seed=42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Sets seed for:
        - NumPy
        - Python's random module
        - PyTorch (CPU and CUDA)
        - Python hash seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"> Random seed set to {seed}")


def fetch_scheduler(optimizer, CFG):
    """Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        CFG: Configuration object with scheduler settings

    Returns:
        Learning rate scheduler instance or None

    Supported schedulers:
        - CosineAnnealingLR
        - CosineAnnealingWarmRestarts
        - ReduceLROnPlateau
        - ExponentialLR
    """
    if CFG.scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, eta_min=CFG.min_lr)
    elif CFG.scheduler == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=CFG.min_lr,
        )
    elif CFG.scheduler == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {CFG.scheduler}")
