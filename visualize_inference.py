import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import argparse

# Ensure project root is in path
sys.path.append(os.getcwd())

from dsb18_core.dataset.utils import rle_decode

def find_image_path(img_id, base_dirs):
    for base in base_dirs:
        path = os.path.join(base, img_id, "images", f"{img_id}.png")
        if os.path.exists(path):
            return path
    return None

def visualize_inference(csv_path, base_dirs, output_dir, num_samples=8, specific_ids=None):
    """
    Visualize predictions from a submission CSV.
    """
    print(f"Loading submission from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: Submission file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    image_ids = df['ImageId'].unique()
    print(f"Found {len(image_ids)} images in submission.")
    
    if specific_ids:
        # Match prefixes if full IDs are not provided
        samples = []
        for sid in specific_ids:
            # Find the full ID in image_ids that starts with sid
            matches = [tid for tid in image_ids if tid.startswith(sid)]
            if matches:
                samples.append(matches[0])
            else:
                print(f"Warning: Could not find image ID starting with {sid}")
    else:
        # Filter out samples with no masks
        counts = df.groupby('ImageId').size().reset_index(name='count')
        images_with_masks = counts[counts['count'] > 0]['ImageId'].values
        
        if len(images_with_masks) > 0:
            samples = images_with_masks[:num_samples]
        else:
            samples = image_ids[:num_samples]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_id in samples:
        img_path = find_image_path(img_id, base_dirs)
        if not img_path:
            print(f"Warning: Could not find image file for ID {img_id}")
            continue
            
        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            
        h, w = img.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        img_df = df[df['ImageId'] == img_id]
        mask_count = 0
        for rle in img_df['EncodedPixels']:
            if pd.isna(rle) or rle == "": continue
            try:
                mask = rle_decode(rle, (h, w))
                combined_mask = np.maximum(combined_mask, mask)
                mask_count += 1
            except: pass
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img)
        axes[0].set_title(f"Original: {img_id[:10]}...")
        axes[0].axis('off')
        
        axes[1].imshow(combined_mask, cmap='hot')
        axes[1].set_title(f"Mask ({mask_count} nuclei)")
        axes[1].axis('off')
        
        # Overlay
        overlay = img.copy()
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_overlay = np.zeros_like(img)
        mask_overlay[combined_mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)
        cv2.drawContours(blended, contours, -1, (255, 255, 0), 1)
        
        axes[2].imshow(blended)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{img_id[:15]}_pred.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DSB 2018 predictions")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--samples", type=int, default=8, help="Number of random samples to visualize")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated list of image IDs or prefixes")
    
    args = parser.parse_args()
    
    specific_ids = args.ids.split(",") if args.ids else None
    base_dirs = ["data/stage1_test", "data/stage2_test_final"]
    
    visualize_inference(
        args.submission, 
        base_dirs, 
        args.output, 
        num_samples=args.samples, 
        specific_ids=specific_ids
    )
