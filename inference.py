from config import CFG
import wandb
import argparse
from tqdm import tqdm

tqdm.pandas()
import os
import torch
import torch.optim as optim
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from dsb18_core.utils import set_seed
from dsb18_core import get_model, get_dataset_mapping
from dsb18_core.pipeline import get_train_valid
from dsb18_core.utils import fetch_scheduler, save_to_submission
from train import get_args

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    _, _, test_loader = get_dataset_mapping(args)

    # Build model
    print(f"Loading model: {args.net_structure}")
    model = get_model(args)
    model.load_state_dict(
        torch.load(
            args.checkpoint_path,
            map_location=torch.device(args.device),
        )
    )
    model.to(args.device)

    # Print fold information
    print(f"\n{'#'*35}")
    print(f"Fold: {args.fold_selected}")
    print(f"{'#'*35}\n")

    # Get training function
    _, inference_fn = get_train_valid(args)

    preds_test_upsampled, id_lists = inference_fn(
        model,
        device=args.device,
        dataloader=test_loader,
        CFG=args,
    )

    save_to_submission(
        id_lists, preds_test_upsampled, submission_filename=f"submission_{args.fold_selected}.csv"
    )
