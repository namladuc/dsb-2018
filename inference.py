from config import CFG
import wandb
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from dsb18_core.utils import set_seed
from dsb18_core import get_model, get_dataset_mapping
from dsb18_core.pipeline import get_train_valid
from dsb18_core.utils import fetch_scheduler, save_to_submission
from dsb18_core.dataset.utils import reset_size_pred
from train import get_args

def extract_sample_meta(meta_batch, index):
    """Recursively extract metadata for a single sample from a batched dictionary."""
    sample_meta = {}
    for k, v in meta_batch.items():
        if isinstance(v, dict):
            sample_meta[k] = extract_sample_meta(v, index)
        elif isinstance(v, (list, tuple)):
            # If it's a list/tuple, check if we should index it directly (batch of strings/scalars)
            # or if it's a collection of batched items (like (h_batch, w_batch))
            is_collection_of_batches = any(isinstance(item, (torch.Tensor, np.ndarray, list, tuple)) for item in v)
            
            if not is_collection_of_batches:
                try: sample_meta[k] = v[index]
                except Exception: sample_meta[k] = v
            else:
                # Collection of batched tensors/lists (like (h_batch, w_batch))
                extracted_list = []
                for item in v:
                    if isinstance(item, (torch.Tensor, np.ndarray, list, tuple)):
                        try:
                            val = item[index]
                            if torch.is_tensor(val): val = val.item() if val.numel() == 1 else val.tolist()
                            elif isinstance(val, np.ndarray): val = val.item() if val.size == 1 else val.tolist()
                            extracted_list.append(val)
                        except Exception: extracted_list.append(item)
                    else:
                        extracted_list.append(item)
                sample_meta[k] = tuple(extracted_list) if isinstance(v, tuple) else extracted_list
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            try:
                val = v[index]
                if torch.is_tensor(val): sample_meta[k] = val.item() if val.numel() == 1 else val.tolist()
                elif isinstance(val, np.ndarray): sample_meta[k] = val.item() if val.size == 1 else val.tolist()
                else: sample_meta[k] = val
            except Exception: sample_meta[k] = v
        else:
            sample_meta[k] = v
    return sample_meta

def run_debug_inference(model, test_loader, args):
    model.eval()
    os.makedirs("debug_inference", exist_ok=True)
    
    predictions = []
    id_list = []
    
    print("\n--- Running Debug Inference (Top 5 samples) ---")
    
    count = 0
    max_samples = 5
    
    for i, (images, image_ids, meta_normalizations) in enumerate(test_loader):
        if count >= max_samples: break
        
        images = images.to(args.device, dtype=torch.float)
        with torch.no_grad():
            y_pred = model(images)
            if args.isDeeply: y_pred = y_pred[0]
            y_pred_sigmoid = torch.sigmoid(y_pred)
        
        preds_cpu = y_pred_sigmoid.cpu().numpy()
        imgs_cpu = images.cpu().numpy()
        
        for b in range(len(image_ids)):
            if count >= max_samples: break
            img_id = image_ids[b]
            img_id_short = img_id[:10]
            print(f"  > Processing [{count+1}/{max_samples}]: {img_id_short}...")
            
            try:
                # Step 1: Input visualization
                prep_img = imgs_cpu[b].transpose(1, 2, 0)
                # Denormalize roughly for vis (assuming z-score or min-max)
                prep_img = (prep_img - prep_img.min()) / (prep_img.max() - prep_img.min() + 1e-8)
                plt_prep = (prep_img * 255).astype(np.uint8)
                Image.fromarray(plt_prep).save(f"debug_inference/{img_id_short}_step1_input.png")
                
                # Step 2: Raw Prediction
                raw_mask = preds_cpu[b, 0]
                raw_mask_plt = (raw_mask * 255).astype(np.uint8)
                Image.fromarray(raw_mask_plt).save(f"debug_inference/{img_id_short}_step2_raw_mask.png")
                
                # Step 3: Resize back to original
                sample_meta = extract_sample_meta(meta_normalizations, b)
                
                # IMPORTANT: wrap in [raw_mask] to provide (N, H, W) shape expected by reset_size_pred
                # Also ensure we use the correct (H, W) order: Height=256, Width=320
                restored_mask = reset_size_pred(np.expand_dims(raw_mask, 0), [sample_meta])[0]
                
                # Binary threshold (this is what is used for RLE)
                binary_mask = (restored_mask > 0.5).astype(np.uint8)
                predictions.append(binary_mask)
                id_list.append(img_id)
                
                # Step 4: RLE Round-trip Check
                from dsb18_core.dataset.utils import rle_encode, rle_decode
                
                # Encode the individual masks (instances)
                # Note: In DSB2018, each connected component is an instance. 
                # But here we might just have a semantic mask.
                # If your model predicts semantic mask, we need to label components.
                from skimage.morphology import label
                lab_mask = label(binary_mask)
                
                roundtrip_mask = np.zeros_like(binary_mask)
                for i in range(1, lab_mask.max() + 1):
                    m = (lab_mask == i).astype(np.uint8)
                    rle = rle_encode(m)
                    decoded = rle_decode(rle, m.shape)
                    roundtrip_mask = np.maximum(roundtrip_mask, decoded)
                
                # Step 5: Save visualizations
                if args.debug_vis:
                    os.makedirs("debug_inference", exist_ok=True)
                    # Final Restored Mask
                    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(f"debug_inference/{img_id_short}_step3_final.png")
                    # Decoded from RLE
                    Image.fromarray((roundtrip_mask * 255).astype(np.uint8)).save(f"debug_inference/{img_id_short}_step4_rle_decoded.png")
                    
                    # Step 6: Mask Overlay on Original
                    # Find original image
                    orig_img = None
                    for p in ["data/stage1_test", "data/stage2_test_final", "data/stage1_train"]:
                        p_img = os.path.join(p, img_id, "images", f"{img_id}.png")
                        if os.path.exists(p_img):
                            orig_img = np.array(Image.open(p_img).convert("RGB"))
                            break
                    
                    if orig_img is not None:
                        # Create green overlay
                        overlay = orig_img.copy()
                        overlay[binary_mask > 0] = [0, 255, 0] # Green where mask is
                        # Blend
                        blended = cv2.addWeighted(orig_img, 0.6, overlay, 0.4, 0)
                        Image.fromarray(blended).save(f"debug_inference/{img_id_short}_step5_overlay.png")
                        print(f"    [v] Saved step3, step4, and step5_overlay.png")
                    else:
                        print(f"    [v] Saved step3 and step4 (Overlay skipped: original not found)")
                else:
                    print(f"    [v] RLE Round-trip OK (Images NOT saved, use --debug_vis to save)")
                
                # Step 4: Final Overlay
                orig_path = None
                for p in ["data/stage1_test", "data/stage2_test_final"]:
                    cand = os.path.join(p, img_id, "images", f"{img_id}.png")
                    if os.path.exists(cand):
                        orig_path = cand
                        break
                
                if orig_path:
                    orig_img = np.array(Image.open(orig_path).convert("RGB"))
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)
                    
                    overlay = orig_img.copy()
                    # Apply green mask
                    overlay[binary_mask > 0] = [0, 255, 0]
                    final_vis = cv2.addWeighted(orig_img, 0.7, overlay, 0.3, 0)
                    
                    # Draw yellow contours
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(final_vis, contours, -1, (255, 255, 0), 1)
                    
                    Image.fromarray(final_vis).save(f"debug_inference/{img_id_short}_step3_final.png")
                    print(f"    [v] Saved step3_final.png")
                else:
                    print(f"    [!] Warning: Original image not found for {img_id_short}")
                
                count += 1
            except Exception as e:
                print(f"    [X] Error processing {img_id_short}: {e}")
                
    return predictions, id_list

if __name__ == "__main__":
    from train import get_args
    args = get_args()
    is_showcase = args.showcase
    if args.input_path:
        args.path_data = args.input_path
    
    set_seed(args.seed)
    _, _, test_loader = get_dataset_mapping(args)
    
    print(f"Loading model: {args.net_structure}")
    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(args.device)))
    model.to(args.device)

    if is_showcase:
        preds, id_lists = run_debug_inference(model, test_loader, args)
        submission_filename = "showcase_submission.csv"
    else:
        _, inference_fn = get_train_valid(args)
        preds, id_lists = inference_fn(model, device=args.device, dataloader=test_loader, CFG=args)
        
        if args.output_path:
            submission_filename = args.output_path
        else:
            submission_filename = f"submission_{args.fold_selected}.csv"

    save_to_submission(id_lists, preds, submission_filename=submission_filename)
    print(f"\nDone! Results saved to {submission_filename}")
