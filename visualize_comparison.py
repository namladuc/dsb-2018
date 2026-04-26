import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dsb18_core import get_model
from dsb18_core.pipeline import get_train_valid
from dsb18_core.dataset.utils import rle_decode
from dsb18_core.build_dataset import get_dataset_mapping
from skimage.morphology import label

def visualize_comparison(model_name, checkpoint_path, img_id=None, num_samples=3):
    # Paths
    data_dir = "data"
    solution_path = os.path.join(data_dir, "stage1_solution.csv")
    output_dir = "debug_vis_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(solution_path):
        print(f"Error: {solution_path} not found.")
        return

    solution_df = pd.read_csv(solution_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model and Args (Fake args for loading)
    class Args:
        def __init__(self):
            self.net_structure = model_name.split('-')[0]
            if "PlusPlus" in model_name: self.net_structure = "FusionUnet2DPlusPlus"
            elif "FusionUnet2DPP" in model_name: self.net_structure = "FusionUnet2DPlusPlus"
            
            self.encoder_backbone = 'resnet18'
            if 'resnet152' in model_name: self.encoder_backbone = 'resnet152'
            if 'timmb3' in model_name: self.encoder_backbone = 'timm-efficientnet-b3'
            
            self.encoder_weights = 'imagenet'
            self.s_channel = 24
            self.input_channel = 3
            self.num_classes = 1
            self.isDeeply = False
            if 'deeply' in model_name: self.isDeeply = True
            if 'test03' in model_name: self.isDeeply = True # Special case for user's test03
            if 'test02' in model_name: self.isDeeply = True # Special case for user's test02
            
            self.use_parallel = False
            self.normalization_method = 'min_max'
            self.path_data = data_dir
            self.numWorker = 0
            self.train_bs = 1
            self.valid_bs = 1
            self.aug = 'baseline'
            self.debug = False
            self.test_stage = "stage1"
            self.img_size = (256, 256)
            self.normalization_scope = 'image'
            self.spacing = (1.0, 1.0)
            self.image_interpolation = 3
            self.mask_interpolation = 1
            self.seed = 42
            self.isPinMemory = False
            self.preprocessing_params = {}
            self.dataset_type = "DSB2018"
            self.input_channel = 3
            self.num_classes = 1
            self.use_parallel = False

    try:
        # 1. Load configuration from arg.md if it exists
        model_dir = os.path.dirname(checkpoint_path)
        arg_file = os.path.join(model_dir, "arg.md")
        
        from train import get_args
        import sys
        import shlex
        
        orig_argv = sys.argv
        if os.path.exists(arg_file):
            with open(arg_file, "r") as f:
                extra_args = f.read().strip()
            sys.argv = ["dummy.py"] + shlex.split(extra_args)
            print(f"Loading args from: {arg_file}")
        else:
            sys.argv = ["dummy.py"]
            
        args = get_args()
        sys.argv = orig_argv
        
        # Override with local paths if on Windows/Local
        if not os.path.exists("/kaggle/working/"):
            args.path_data = "data"
        else:
            args.path_data = "/kaggle/working/dsb-data-2018"
            
        # Override with necessary paths/device
        args.device = device
        
        model = get_model(args)
        
        # Load weights
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 2. Setup Data Loader to get preprocessed images
        _, _, test_loader = get_dataset_mapping(args)
    except Exception as e:
        print(f"CRITICAL ERROR during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Pick samples
    available_ids = [d for d in os.listdir(os.path.join(data_dir, "stage1_test")) if os.path.isdir(os.path.join(data_dir, "stage1_test", d))]
    solution_ids = solution_df["ImageId"].unique()
    valid_ids = [i for i in available_ids if i in solution_ids]
    
    if img_id and img_id in valid_ids:
        target_ids = [img_id]
    else:
        target_ids = valid_ids[:num_samples]
        
    print(f"Visualizing {len(target_ids)} samples...")

    for target_id in target_ids:
        # Get Image
        img_path = os.path.join(data_dir, "stage1_test", target_id, "images", f"{target_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Get GT
        gt_rows = solution_df[solution_df["ImageId"] == target_id]
        gt_instances = []
        gt_mask_merged = np.zeros((h, w), dtype=np.uint8)
        for _, row in gt_rows.iterrows():
            rle = row["EncodedPixels"]
            if pd.isna(rle): continue
            inst = rle_decode(rle, (h, w))
            gt_instances.append(inst)
            gt_mask_merged[inst > 0] = 1
            
        # Get Prediction
        # We find the image in the loader to use the preprocessing
        found_data = None
        for batch in test_loader:
            images, ids, metas_batched = batch
            if target_id in ids:
                idx = ids.index(target_id)
                # Unbatch first to get list of dicts
                from dsb18_core.dataset.utils import unbatch_meta
                metas_list = unbatch_meta(metas_batched)
                found_data = (images[idx:idx+1], metas_list[idx])
                break
        
        if found_data:
            batch_img, single_meta = found_data
            with torch.no_grad():
                pred = model(batch_img.to(device))
                if args.isDeeply: pred = pred[0]
                pred_prob = torch.sigmoid(pred).cpu().numpy().squeeze()
            
            # Use runner_2d logic to reset size (center-crop reversal)
            from dsb18_core.dataset.utils import reset_size_pred
            pred_upsampled = reset_size_pred(np.expand_dims(pred_prob, 0), [single_meta])[0]
            pred_binary = (pred_upsampled > 0.5).astype(np.uint8)
            pred_labeled = label(pred_binary)
        else:
            print(f"Warning: Could not find {target_id} in loader")
            continue

        # Calculate Metrics for this specific image
        from dsb18_core.metrics import dsb2018_map, dice_numpy
        # GT needs to be labeled or a list of masks. Let's use labeled for simplicity
        gt_labeled = label(gt_mask_merged)
        
        m_ap = dsb2018_map(gt_labeled, pred_labeled)
        d_score = dice_numpy(gt_mask_merged, pred_binary)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Col 1: GT Overlay
        vis_gt = img.copy()
        vis_gt[gt_mask_merged > 0] = vis_gt[gt_mask_merged > 0] * 0.5 + np.array([0, 255, 0]) * 0.5 # Green
        axes[0].imshow(vis_gt)
        axes[0].set_title(f"Ground Truth (Green)\nID: {target_id[:10]}...")
        axes[0].axis('off')
        
        # Col 2: Pred Overlay
        vis_pred = img.copy()
        vis_pred[pred_binary > 0] = vis_pred[pred_binary > 0] * 0.5 + np.array([255, 0, 0]) * 0.5 # Red
        axes[1].imshow(vis_pred)
        axes[1].set_title(f"Prediction (Red)\nmAP: {m_ap:.4f} | Dice: {d_score:.4f}")
        axes[1].axis('off')
        
        # Col 3: Comparison
        vis_comp = img.copy()
        # TP: Yellow (Red + Green)
        tp = (gt_mask_merged > 0) & (pred_binary > 0)
        # FN: Blue (Missing)
        fn = (gt_mask_merged > 0) & (pred_binary == 0)
        # FP: Magenta (Extra)
        fp = (gt_mask_merged == 0) & (pred_binary > 0)
        
        vis_comp[tp] = vis_comp[tp] * 0.5 + np.array([255, 255, 0]) * 0.5 # Yellow
        vis_comp[fn] = vis_comp[fn] * 0.5 + np.array([0, 0, 255]) * 0.5 # Blue
        vis_comp[fp] = vis_comp[fp] * 0.5 + np.array([255, 0, 255]) * 0.5 # Magenta
        
        axes[2].imshow(vis_comp)
        axes[2].set_title("Comparison\nYellow: TP, Blue: FN, Magenta: FP")
        axes[2].axis('off')
        
        save_path = os.path.join(output_dir, f"comp_{model_name}_{target_id}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    # Analyzing why Baseline is so low
    model_name = "Unet2D-Baseline_baseline"
    checkpoint = "data/model_checkpoint/Unet2D-Baseline_baseline/best_epoch_Unet2D-Baseline_00.bin"
    visualize_comparison(model_name, checkpoint)
