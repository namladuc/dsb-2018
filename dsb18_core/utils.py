import os
import torch
import random
import numpy as np
import pandas as pd
from skimage.morphology import label
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


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def save_to_submission(test_ids, preds_test_upsampled, submission_filename="submission.csv"):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub["ImageId"] = new_test_ids
    sub["EncodedPixels"] = pd.Series(rles).apply(lambda x: " ".join(str(y) for y in x))
    sub.to_csv(submission_filename, index=False)
