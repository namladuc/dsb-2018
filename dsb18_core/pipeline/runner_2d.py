from dsb18_core import *
import wandb
import numpy as np
from tqdm import tqdm

tqdm.pandas()
import gc
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import os
from datetime import datetime
from ..loss import criterion
from ..metrics import get_metric


# ==================== Training Utilities ====================


def compute_metrics(y_true, y_pred, CFG, metrics_list):
    """Compute multiple evaluation metrics.

    Args:
        y_true: Ground truth masks (B, H, W)
        y_pred: Model predictions (B, H, W)
        CFG: Configuration object
        metrics_list: List of metric names to compute

    Returns:
        Dictionary with metric names and values
    """
    metrics_values = {}
    for metric_name in metrics_list:
        try:
            metric_fn = get_metric(metric_name)
            value = metric_fn(y_true, y_pred, CFG).cpu().detach().numpy()
            metrics_values[metric_name] = value
        except Exception as e:
            print(f"Warning: Could not compute metric {metric_name}: {e}")
    return metrics_values


# ==================== Training --==================


def train_one_epoch2d(model, optimizer, scheduler, dataloader, device, CFG):
    """Train model for one epoch.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        dataloader: Training data loader
        device: Device to train on (cuda/cpu)
        CFG: Configuration object

    Returns:
        Average loss for the epoch
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train ")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        batch_size = images.size(0)

        optimizer.zero_grad()
        y_pred = model(images)

        # Handle deeply supervised models with multiple predictions
        if CFG.isDeeply:
            loss = sum(criterion(p, masks, CFG) for p in y_pred) / len(y_pred)
        else:
            loss = criterion(y_pred, masks, CFG)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
        )

        if CFG.debug:
            break

    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch2d(model, dataloader, device, optimizer, CFG):
    """Validate model for one epoch.

    Args:
        model: PyTorch model in eval mode
        dataloader: Validation data loader
        device: Device to validate on
        optimizer: Optimizer (for learning rate logging)
        CFG: Configuration object

    Returns:
        Tuple of (epoch_loss, metrics_dict)
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    metrics_list = getattr(CFG, "metrics", ["dice", "iou"])
    metrics_tracker = defaultdict(list)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        batch_size = images.size(0)

        y_preds = model(images)

        # Handle deeply supervised models
        y_pred = y_preds[0] if CFG.isDeeply else y_preds
        loss = criterion(y_pred, masks, CFG)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        # Compute metrics
        y_pred_sigmoid = nn.Sigmoid()(y_pred)
        metrics_values = compute_metrics(masks, y_pred_sigmoid, CFG, metrics_list)

        for metric_name, metric_value in metrics_values.items():
            metrics_tracker[metric_name].append(metric_value)

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
        )

        if CFG.debug:
            print(f"  Input shape: {images.shape}, Output shape: {y_pred.shape}")
            break

    avg_metrics = {k: np.mean(v) for k, v in metrics_tracker.items()}
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, avg_metrics


def run_training2d(model, optimizer, scheduler, run, num_epochs, train_loader, valid_loader, CFG):
    """Training pipeline for 2D segmentation models.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        run: WandB run object (can be None)
        num_epochs: Total number of epochs to train
        train_loader: Training dataloader
        valid_loader: Validation dataloader
        CFG: Configuration object with training parameters

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Setup experiment directory
    base_dir = os.path.join(os.getcwd(), "runs")
    exp_name = (
        f"{CFG.dataset}_{CFG.net_structure}_{CFG.encoder_backbone}_{CFG.model_name}_{CFG.aug}"
    )
    exp_dir = os.path.join(base_dir, exp_name, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    os.makedirs(exp_dir, exist_ok=True)

    # Get metrics configuration
    metrics_list = getattr(CFG, "metrics", ["dice", "iou"])
    primary_metric = metrics_list[0] if metrics_list else "dice"

    if CFG.using_wandb == 1:
        # To automatically log gradients
        wandb.watch(model, log_freq=100)

    # Resume scheduler if resuming from checkpoint
    if CFG.resume_train:
        for _ in range(CFG.epochs_res):
            scheduler.step()
    else:
        CFG.epochs_res = 1

    # Print device info
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}\n")

    # Print training configuration
    print(f"\n{'='*80}")
    print("Training Configuration:")
    print(f"  Loss: {getattr(CFG, 'loss_name', 'dice_entropy')}")
    print(f"  Metrics: {metrics_list}")
    print(f"  Primary Metric: {primary_metric}")
    print(f"{'='*80}\n")

    # Initialize training state
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric_value = -np.inf
    best_epoch = -1

    if CFG.resume_train:
        best_metric_value = getattr(CFG, f"best_{primary_metric}", -1)
        best_epoch = CFG.best_epoch

    history = defaultdict(list)

    # Training loop
    for epoch in range(CFG.epochs_res, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}", end=" | ")

        # Train and validate
        train_loss = train_one_epoch2d(
            model, optimizer, scheduler, dataloader=train_loader, device=CFG.device, CFG=CFG
        )

        val_loss, val_metrics = valid_one_epoch2d(
            model, valid_loader, device=CFG.device, optimizer=optimizer, CFG=CFG
        )

        scheduler.step()

        if CFG.debug:
            break

        # Update history
        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        for metric_name, metric_value in val_metrics.items():
            history[f"Valid {metric_name.capitalize()}"].append(metric_value)

        # Log results
        log_str = f"Loss: {val_loss:0.4f}"
        for metric_name, metric_value in val_metrics.items():
            log_str += f" | {metric_name.upper()}: {metric_value:0.4f}"
        print(log_str)

        if CFG.using_wandb == 1:
            # Log the metrics
            log_dict = {
                "Train Loss": train_loss,
                "Valid Loss": val_loss,
                "LR": scheduler.get_last_lr()[0],
            }
            for metric_name, metric_value in val_metrics.items():
                log_dict[f"Valid {metric_name.capitalize()}"] = metric_value
            wandb.log(log_dict)

        # Check if primary metric improved
        current_metric = val_metrics.get(primary_metric, -np.inf)
        if current_metric > best_metric_value:
            print(
                f"✓ {primary_metric.upper()} improved "
                f"({best_metric_value:0.4f} → {current_metric:0.4f})"
            )
            best_metric_value = current_metric
            best_epoch = epoch

            if CFG.using_wandb == 1:
                run.summary[f"Best {primary_metric}"] = best_metric_value
                run.summary["Best Epoch"] = best_epoch

            best_model_wts = copy.deepcopy(model.state_dict())
            best_path = os.path.join(
                exp_dir, f"best_epoch_{CFG.model_name}_{CFG.fold_selected:02d}.bin"
            )
            torch.save(model.state_dict(), best_path)
            if CFG.using_wandb:
                wandb.save(best_path, base_path=exp_dir)

        # Save last checkpoint
        last_path = os.path.join(
            exp_dir, f"last_epoch_{CFG.model_name}_{CFG.fold_selected:02d}.bin"
        )
        torch.save(model.state_dict(), last_path)
        if CFG.using_wandb:
            wandb.save(last_path, base_path=exp_dir)

        print()

    # Print summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\nTraining complete in {hours}h {minutes}m {seconds}s")
    print(f"Best {primary_metric.upper()}: {best_metric_value:.4f} (Epoch {best_epoch})")

    # Load best model
    model.load_state_dict(best_model_wts)

    return model, history
