import os
import torch
import pandas as pd
import numpy as np
import shlex
from tqdm import tqdm
from skimage.morphology import label

from config import CFG
from dsb18_core import get_model, get_dataset_mapping
from dsb18_core.pipeline import get_train_valid
from dsb18_core.utils import save_to_submission
from dsb18_core.metrics import dsb2018_map, dice_numpy
from dsb18_core.dataset.utils import rle_decode

def evaluate_models():
    # Configuration for paths (local paths for debug, but script works on Kaggle too)
    checkpoint_dir = "data/model_checkpoint"
    data_dir = "data"
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    solution_path = os.path.join(data_dir, "stage1_solution.csv")
    if not os.path.exists(solution_path):
        print(f"Warning: {solution_path} not found. Evaluation will be skipped.")
        solution_df = None
    else:
        print(f"Found solution file: {solution_path}")
        solution_df = pd.read_csv(solution_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models_to_run = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    results = []

    for model_name in models_to_run:
        model_path = os.path.join(checkpoint_dir, model_name)
        arg_file = os.path.join(model_path, "arg.md")
        
        if not os.path.exists(arg_file):
            print(f"Skipping {model_name}: arg.md missing")
            continue

        # Find checkpoint
        checkpoint_file = None
        for f in os.listdir(model_path):
            if f.endswith(".bin") or f.endswith(".pth"):
                checkpoint_file = os.path.join(model_path, f)
                break
        
        if not checkpoint_file:
            print(f"Skipping {model_name}: checkpoint missing")
            continue

        print(f"\n" + "="*60)
        print(f"Processing Model: {model_name}")
        print("="*60)

        # 1. Parse arguments from arg.md
        with open(arg_file, "r") as f:
            extra_args_str = f.read().strip()
        
        # Use a temporary parser to update CFG
        import argparse
        from train import get_args
        # We need to hack sys.argv to use get_args() easily
        import sys
        original_argv = sys.argv
        sys.argv = ["dummy.py"] + shlex.split(extra_args_str)
        args = get_args()
        sys.argv = original_argv
        
        # Override paths for local/Kaggle consistency
        args.path_data = data_dir
        args.device = device
        args.numWorker = 0 # Safety for local run
        
        # 2. Build Dataset
        _, _, test_loader = get_dataset_mapping(args)
        
        # 3. Load Model
        model = get_model(args)
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        model.to(device)
        
        # 4. Run Inference
        _, inference_fn = get_train_valid(args)
        predictions, id_list = inference_fn(model, device=device, dataloader=test_loader, CFG=args)
        
        # 5. Save Submission
        sub_path = os.path.join(output_dir, f"submission_{model_name}.csv")
        save_to_submission(predictions, id_list, sub_path)
        print(f"Generated submission: {sub_path}")

        # 6. Evaluate if solution exists
        if solution_df is not None:
            print(f"Calculating metrics for Stage 1 Test...")
            # Filter predictions for IDs that are in solution_df
            solution_ids = set(solution_df["ImageId"].unique())
            
            map_scores = []
            dice_scores = []
            
            # Map predictions to ID for easier lookup
            pred_map = {img_id: mask for img_id, mask in zip(id_list, predictions)}
            
            for img_id in tqdm(solution_ids, desc="Evaluating"):
                if img_id not in pred_map:
                    continue
                
                # Get Pred
                pred_mask = pred_map[img_id]
                
                # Get GT (Solution CSV has multiple rows per image, one per instance)
                gt_rows = solution_df[solution_df["ImageId"] == img_id]
                h, w = pred_mask.shape
                gt_instances = []
                for _, row in gt_rows.iterrows():
                    rle = row["EncodedPixels"]
                    if pd.isna(rle): continue
                    inst = rle_decode(rle, (h, w))
                    gt_instances.append(inst)
                
                if not gt_instances:
                    gt_instances = [np.zeros((h, w), dtype=np.uint8)]
                
                # mAP calculation
                score_map = dsb2018_map(gt_instances, pred_mask)
                map_scores.append(score_map)
                
                # Dice calculation (semantic)
                combined_gt = np.max(gt_instances, axis=0) if len(gt_instances) > 0 else np.zeros((h, w))
                combined_pred = (pred_mask > 0).astype(np.uint8)
                score_dice = dice_numpy(combined_gt, combined_pred)
                dice_scores.append(score_dice)
            
            mean_map = np.mean(map_scores) if map_scores else 0
            mean_dice = np.mean(dice_scores) if dice_scores else 0
            
            print(f"Result for {model_name}: mAP={mean_map:.4f}, Dice={mean_dice:.4f}")
            results.append({
                "Model": model_name,
                "mAP": mean_map,
                "Dice": mean_dice,
                "Submission": sub_path
            })

    # Final Report
    if results:
        print("\n" + "#"*60)
        print("FINAL EVALUATION REPORT")
        print("#"*60)
        report_df = pd.DataFrame(results)
        print(report_df.to_string(index=False))
        report_df.to_csv(os.path.join(output_dir, "evaluation_report.csv"), index=False)

if __name__ == "__main__":
    evaluate_models()
