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
from dsb18_core.utils import fetch_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net 2D Segmentation Training")
    # Add arguments for each attribute in the CFG class
    parser.add_argument('--seed', type=int, default=CFG.seed, help="Random seed for reproducibility.")
    parser.add_argument('--debug', type=int, default=CFG.debug, help="Debug mode: saves first 5 images before/after preprocessing to debug/ folder ")
    parser.add_argument('--using_wandb', type=int, default=CFG.using_wandb, help="Enable wandb logging.")
    parser.add_argument('--resume_train', type=bool, default=CFG.resume_train, help="Resume training from a checkpoint.")
    parser.add_argument('--id_wandb', type=str, default=CFG.id_wandb, help="Wandb id for resuming training.")
    parser.add_argument('--checkP_name', type=str, default=CFG.checkP_name, help="Checkpoint name for resuming training.")
    parser.add_argument('--epochs_res', type=int, default=CFG.epochs_res, help="Epochs for resuming training.")
    parser.add_argument('--best_dice', type=float, default=CFG.best_dice, help="Best dice score for resuming training.")
    parser.add_argument('--best_epoch', type=int, default=CFG.best_epoch, help="Best epoch for resuming training.")
    
    # Dataset
    parser.add_argument('--path_data', type=str, default=CFG.path_data, help="Dataset Path (DSB-2018).")
    parser.add_argument('--dataset', type=str, default=CFG.dataset, help="Dataset name (DSB2018).")
    parser.add_argument('--aug', type=str, default=CFG.aug, help="Data augmentation kit set choice.")
    parser.add_argument('--isPinMemory', type=bool, default=CFG.isPinMemory, help="Enable pinned memory if available.")
    parser.add_argument('--numWorker', type=int, default=CFG.numWorker, help="Number of data loader workers.")
    parser.add_argument('--train_bs', type=int, default=CFG.train_bs, help="Batch size for training.")
    parser.add_argument('--valid_bs', type=int, default=CFG.valid_bs, help="Batch size for validation.")
    parser.add_argument('--img_size', type=tuple, default=CFG.img_size, help="Image size (width, height).")
    parser.add_argument('--resize_mode', type=str, default=CFG.resize_mode, help="Resize mode: 'resize_only' or 'pad_and_resize'.")
    parser.add_argument('--spacing', type=tuple, default=CFG.spacing, help="Image spacing (x, y).")
    parser.add_argument('--normalization_method', type=str, default=CFG.normalization_method, help="Intensity normalization method (zscore, percentile, minmax).")
    parser.add_argument('--normalization_scope', type=str, default=CFG.normalization_scope, help="Normalization scope (global or local).")
    parser.add_argument('--image_interpolation', type=int, default=CFG.image_interpolation, help="Image interpolation order (0-5).")
    parser.add_argument('--mask_interpolation', type=int, default=CFG.mask_interpolation, help="Mask interpolation order (0-5).")
    
    # Model
    parser.add_argument('--model_name', type=str, default=CFG.model_name, help="Name of the model.")
    parser.add_argument('--isDeeply', type=bool, default=CFG.isDeeply, help="Enable deep learning features.")
    parser.add_argument('--backbone', type=str, default=CFG.backbone, help="Backbone architecture for the model.")
    parser.add_argument('--epochs', type=int, default=CFG.epochs, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=CFG.lr, help="Learning rate.")
    parser.add_argument('--scheduler', type=str, default=CFG.scheduler, help="Learning rate scheduler type.")
    parser.add_argument('--min_lr', type=float, default=CFG.min_lr, help="Minimum learning rate.")
    parser.add_argument('--T_max', type=int, default=CFG.T_max, help="Maximum number of iterations for the cosine annealing scheduler.")
    parser.add_argument('--T_0', type=int, default=CFG.T_0, help="Number of iterations for a restart in the cosine annealing scheduler.")
    parser.add_argument('--s_channel', type=int, default=CFG.s_channel, help="Number of channels in the first layer of the model.")
    parser.add_argument('--use_parallel', type=bool, default=CFG.use_parallel, help="Use parallel GPU.")
    parser.add_argument('--warmup_epochs', type=int, default=CFG.warmup_epochs, help="Number of warm-up epochs.")
    parser.add_argument('--wd', type=float, default=CFG.wd, help="Weight decay.")
    parser.add_argument('--n_accumulate', type=int, default=CFG.n_accumulate, help="Number of batches to accumulate gradients before a backward/update pass.")
    parser.add_argument('--bilinear', type=bool, default=CFG.bilinear, help="Use bilinear interpolation.")
    parser.add_argument('--device', type=str, default=CFG.device, help="Device for training (cuda:0 for GPU, cpu for CPU).")
    parser.add_argument('--net_structure', type=str, default=CFG.net_structure, help="Network structure type.")
    parser.add_argument('--valid_epochs', type=int, default=CFG.valid_epochs, help="Valid Epoch step")
    parser.add_argument('--input_channel', type=int, default=CFG.input_channel, help="Number of input channels.")
    parser.add_argument('--num_classes', type=int, default=CFG.num_classes, help="Number of output classes.")

    # Loss and Metrics
    parser.add_argument('--loss_name', type=str, default=CFG.loss_name, 
                       help="Loss function: 'dice_entropy', 'tversky_entropy', 'focal_dice', 'bce', 'dice', 'tversky'")
    parser.add_argument('--metrics', type=str, nargs='+', default=CFG.metrics,
                       help="Metrics to track: 'dice', 'iou'")
    parser.add_argument('--fold_selected', type=int, default=CFG.fold_selected,
                       help="Fold index for cross-validation.")
    
    # Fire module config
    parser.add_argument('--expand_ratio', type=int, default=CFG.expand_ratio, help="Expansion ratio in the Fire module.")
    parser.add_argument('--expand_kernel', type=int, default=CFG.expand_kernel, help="Expansion kernel size in the Fire module.")
    
    parser.add_argument('--block_num', type=int, default=CFG.block_num, help="Number of Block In transformers")
    parser.add_argument('--patch_dim', type=int, default=CFG.patch_dim, help="Patch Size Dimension in Transformer")   
    parser.add_argument('--head_num', type=int, default=CFG.head_num, help="Number Of Head in Transformer")
    args = parser.parse_args()
    
    # Initialize WandB if enabled
    if args.using_wandb:
        try:
            if (args.debug):
                api_key = ''
            else:
                api_key = os.environ['WANDB_API_KEY']
            wandb.login(key=api_key)
            anonymous = None
        except:
            anonymous = "must"
    
    set_seed(args.seed)
    train_loader, valid_loader = get_dataset_mapping(args)
    
    # Build model
    print(f"Loading model: {args.net_structure}")
    model = get_model(args)
    
    if args.resume_train:
        print(f"Resuming from checkpoint: {args.checkP_name}")
        model.load_state_dict(torch.load(args.checkP_name))
    
    model.to(args.device)
    
    # Print fold information
    print(f"\n{'#'*35}")
    print(f"Fold: {args.fold_selected}")
    print(f"{'#'*35}\n")
    
    # Get training function
    train_valid_fn = get_train_valid(args)
    
    # Initialize WandB run
    run = None
    if args.using_wandb:
        run = wandb.init(
            project="DSB-ChuyenDe4-MasterDS",
            config={k: v for k, v in vars(args).items() if not k.startswith('_')},
            name=f"{args.model_name}_{args.aug}",
            id=args.id_wandb if args.resume_train else None,
            resume="must" if args.resume_train else "never"
        )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = fetch_scheduler(optimizer, args)
    
    # Train model
    model, history = train_valid_fn(
        model, optimizer, scheduler, run,
        num_epochs=args.epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        CFG=args
    )
    
    if args.using_wandb:
        run.finish()