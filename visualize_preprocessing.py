import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import torch

# Import from project core
# Note: Ensure you are in the project root to run this
from dsb18_core.dataset.preprocessing import crop_background, resample_image, normalize_intensity
from config import CFG

def visualize_steps(image_id=None, path_data="./data"):
    """
    Visualize each step of the DSB 2018 preprocessing pipeline.
    """
    # 1. Setup paths
    train_path = os.path.join(path_data, "stage1_train")
    if image_id is None:
        # Get the first available image
        image_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        if not image_ids:
            print(f"No images found in {train_path}")
            return
        image_id = image_ids[0]
    
    image_file = os.path.join(train_path, image_id, "images", f"{image_id}.png")
    masks_dir = os.path.join(train_path, image_id, "masks")
    
    # 2. Load Original Data
    img_orig = np.array(Image.open(image_file))
    if len(img_orig.shape) == 2:
        img_orig = np.stack([img_orig] * 3, axis=-1)
    elif img_orig.shape[2] == 4:
        img_orig = img_orig[:, :, :3]
        
    mask_files = glob(os.path.join(masks_dir, "*.png"))
    h, w = img_orig.shape[:2]
    mask_orig = np.zeros((h, w), dtype=np.float32)
    for m_file in mask_files:
        m = np.array(Image.open(m_file))
        mask_orig = np.maximum(mask_orig, (m > 0).astype(np.float32))

    # 3. Preprocessing Steps
    steps_images = []
    steps_masks = []
    step_names = []

    # Step 0: Original
    steps_images.append(img_orig)
    steps_masks.append(mask_orig)
    step_names.append(f"Original\n({h}x{w})")

    # Step 1: Cropping
    img_cropped, mask_cropped, crop_info = crop_background(img_orig, mask_orig, threshold=0, margin=10)
    steps_images.append(img_cropped)
    steps_masks.append(mask_cropped)
    step_names.append(f"Cropped\n({img_cropped.shape[0]}x{img_cropped.shape[1]})")

    # Step 2: Resampling (Resize or Pad+Resize)
    target_size = (320, 256) # CFG.img_size (width, height)
    img_resampled, mask_resampled = resample_image(
        img_cropped, 
        target_size, 
        method="bicubic", 
        mask=mask_cropped, 
        resize_mode="pad_and_resize"
    )
    steps_images.append(img_resampled)
    steps_masks.append(mask_resampled)
    step_names.append(f"Resampled\n({target_size[1]}x{target_size[0]})")

    # Step 3: Normalization (Z-Score as example)
    # Using dummy mean/std if fingerprint not run, or use global ones
    img_norm = normalize_intensity(img_resampled, method="z_score", params={"mean": 128.0, "std": 50.0})
    
    # For visualization, normalize back to [0, 1]
    img_norm_vis = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    steps_images.append(img_norm_vis)
    steps_masks.append(mask_resampled)
    step_names.append("Normalized\n(Z-Score)")

    # 4. Plotting
    n_steps = len(step_names)
    fig, axes = plt.subplots(2, n_steps, figsize=(15, 8))
    
    for i in range(n_steps):
        # Image row
        ax_img = axes[0, i]
        ax_img.imshow(steps_images[i].astype(np.uint8) if steps_images[i].max() > 1.0 else steps_images[i])
        ax_img.set_title(step_names[i], fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        # Mask row
        ax_mask = axes[1, i]
        ax_mask.imshow(steps_masks[i], cmap='gray')
        ax_mask.axis('off')
        if i == 0:
            ax_img.set_ylabel("Image", fontsize=12, labelpad=20)
            ax_mask.set_ylabel("Mask", fontsize=12, labelpad=20)

    plt.tight_layout()
    plt.suptitle(f"Preprocessing Pipeline Visualization: {image_id}", fontsize=16, y=1.05)
    
    # Save the result
    output_png = "preprocessing_visualization.png"
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    visualize_steps()
