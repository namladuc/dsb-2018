from config import CFG
import argparse
from tqdm import tqdm

tqdm.pandas()
import torch
import warnings

warnings.filterwarnings("ignore")

from dsb18_core import get_model


def get_args():
    parser = argparse.ArgumentParser(description="U-Net 2D Converted Model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=CFG.checkpoint_path,
        help="Checkpoint path for resuming training.",
    )
    parser.add_argument(
        "--img_size", type=tuple, default=CFG.img_size, help="Image size (width, height)."
    )

    # Model
    parser.add_argument("--model_name", type=str, default=CFG.model_name, help="Name of the model.")
    parser.add_argument(
        "--isDeeply", type=bool, default=CFG.isDeeply, help="Enable deep learning features."
    )
    parser.add_argument(
        "--encoder_backbone",
        type=str,
        default=CFG.encoder_backbone,
        help="Backbone architecture for the model.",
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default=CFG.encoder_weights,
        help="Pretrained weights for the encoder.",
    )
    parser.add_argument("--epochs", type=int, default=CFG.epochs, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=CFG.lr, help="Learning rate.")
    parser.add_argument(
        "--scheduler", type=str, default=CFG.scheduler, help="Learning rate scheduler type."
    )
    parser.add_argument("--min_lr", type=float, default=CFG.min_lr, help="Minimum learning rate.")
    parser.add_argument(
        "--T_max",
        type=int,
        default=CFG.T_max,
        help="Maximum number of iterations for the cosine annealing scheduler.",
    )
    parser.add_argument(
        "--T_0",
        type=int,
        default=CFG.T_0,
        help="Number of iterations for a restart in the cosine annealing scheduler.",
    )
    parser.add_argument(
        "--s_channel",
        type=int,
        default=CFG.s_channel,
        help="Number of channels in the first layer of the model.",
    )
    parser.add_argument(
        "--use_parallel", type=bool, default=CFG.use_parallel, help="Use parallel GPU."
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=CFG.warmup_epochs, help="Number of warm-up epochs."
    )
    parser.add_argument("--wd", type=float, default=CFG.wd, help="Weight decay.")
    parser.add_argument(
        "--n_accumulate",
        type=int,
        default=CFG.n_accumulate,
        help="Number of batches to accumulate gradients before a backward/update pass.",
    )
    parser.add_argument(
        "--bilinear", type=bool, default=CFG.bilinear, help="Use bilinear interpolation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CFG.device,
        help="Device for training (cuda:0 for GPU, cpu for CPU).",
    )
    parser.add_argument(
        "--net_structure", type=str, default=CFG.net_structure, help="Network structure type."
    )
    parser.add_argument(
        "--valid_epochs", type=int, default=CFG.valid_epochs, help="Valid Epoch step"
    )
    parser.add_argument(
        "--input_channel", type=int, default=CFG.input_channel, help="Number of input channels."
    )
    parser.add_argument(
        "--num_classes", type=int, default=CFG.num_classes, help="Number of output classes."
    )

    # Loss and Metrics
    parser.add_argument(
        "--loss_name",
        type=str,
        default=CFG.loss_name,
        help="Loss function: 'dice_entropy', 'tversky_entropy', 'focal_dice', 'bce', 'dice', 'tversky'",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=CFG.metrics,
        help="Metrics to track: 'dice', 'iou'",
    )
    parser.add_argument(
        "--fold_selected",
        type=int,
        default=CFG.fold_selected,
        help="Fold index for cross-validation.",
    )

    # Fire module config
    parser.add_argument(
        "--expand_ratio",
        type=int,
        default=CFG.expand_ratio,
        help="Expansion ratio in the Fire module.",
    )
    parser.add_argument(
        "--expand_kernel",
        type=int,
        default=CFG.expand_kernel,
        help="Expansion kernel size in the Fire module.",
    )

    parser.add_argument(
        "--block_num", type=int, default=CFG.block_num, help="Number of Block In transformers"
    )
    parser.add_argument(
        "--patch_dim", type=int, default=CFG.patch_dim, help="Patch Size Dimension in Transformer"
    )
    parser.add_argument(
        "--head_num", type=int, default=CFG.head_num, help="Number Of Head in Transformer"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Build model
    print(f"Loading model: {args.net_structure}")
    model = get_model(args)

    print(f"Resuming from checkpoint: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))

    model.to(args.device)
    model.eval()

    example_inputs = torch.randn(1, args.input_channel, *args.img_size).to(args.device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_inputs,
            "segmentation_model_timm.onnx",
            input_names=["input"],
            output_names=["output1", "output2", "output3", "output4", "output5"],
            opset_version=17,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
