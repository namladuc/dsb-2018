import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import albumentations as A

# Import from project core
from dsb18_core.dataset.preprocessing import preprocess_pipeline
from dsb18_core.dataset.aug import get_aug_dict

class DummyCFG:
    aug = "aug1" # We visualize aug1 since baseline is empty
    img_size = (320, 256)
    normalization_method = "z_score"
    normalization_scope = "global"
    spacing = (1, 1)
    image_interpolation = 3
    mask_interpolation = 1
    crop_background = True
    crop_threshold = 0
    crop_margin = 10
    resize_mode = "pad_and_resize"

def visualize_augmentations(num_samples=5, path_data="./data"):
    # 1. Setup paths and config
    CFG = DummyCFG()
    train_path = os.path.join(path_data, "stage1_train")
    image_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    image_id = image_ids[0]
    
    image_file = os.path.join(train_path, image_id, "images", f"{image_id}.png")
    masks_dir = os.path.join(train_path, image_id, "masks")
    
    # 2. Load and Preprocess
    img_orig = np.array(Image.open(image_file))
    if len(img_orig.shape) == 2:
        img_orig = np.stack([img_orig] * 3, axis=-1)
    elif img_orig.shape[2] == 4:
        img_orig = img_orig[:, :, :3]
        
    mask_files = glob(os.path.join(masks_dir, "*.png"))
    mask_orig = np.zeros(img_orig.shape[:2], dtype=np.float32)
    for m_file in mask_files:
        mask_orig = np.maximum(mask_orig, (np.array(Image.open(m_file)) > 0))

    pp_config = {
        "crop_background": True,
        "target_size": (320, 256),
        "resize_mode": "pad_and_resize",
        "intensity_normalization": {"method": "min_max"}
    }
    img_pre, mask_pre, _ = preprocess_pipeline(img_orig, mask_orig, pp_config)
    
    # 3. Define Individual Transforms to Visualize
    aug_list = [
        ("Original", None),
        ("CLAHE", A.CLAHE(p=1.0)),
        ("Sharpen", A.Sharpen(p=1.0)),
        ("GaussNoise", A.GaussNoise(p=1.0)),
        ("Blur", A.Blur(blur_limit=7, p=1.0)),
        ("BrightnessContrast", A.RandomBrightnessContrast(p=1.0)),
        ("ShiftScaleRotate", A.ShiftScaleRotate(p=1.0)),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("HueSaturation", A.HueSaturationValue(p=1.0)),
        ("ChannelShuffle", A.ChannelShuffle(p=1.0)),
    ]
    
    # 4. Plot
    n = len(aug_list)
    rows = 2
    cols = (n + 1) // 2
    fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 12))
    axes = axes.flatten()

    for i, (name, aug) in enumerate(aug_list):
        if aug is not None:
            augmented = aug(image=img_pre, mask=mask_pre)
            img_show = augmented['image']
            mask_show = augmented['mask']
        else:
            img_show = img_pre
            mask_show = mask_pre

        # Plot Image
        ax_idx = (i // cols) * (cols * 2) + (i % cols)
        axes[ax_idx].imshow(img_show)
        axes[ax_idx].set_title(name, fontsize=14, fontweight='bold')
        axes[ax_idx].axis('off')

        # Plot Mask below it
        ax_mask_idx = ax_idx + cols
        axes[ax_mask_idx].imshow(mask_show, cmap='gray')
        axes[ax_mask_idx].axis('off')

    # Remove extra empty subplots
    for j in range(len(axes)):
        if not axes[j].get_visible():
            continue
        # If it's in the second or fourth row but beyond the list length
        row_idx = j // cols
        col_idx = j % cols
        item_idx = (row_idx // 2) * cols + col_idx
        if item_idx >= n:
            axes[j].axis('off')
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle("Individual Augmentation Methods Visualization", fontsize=20, y=1.02)
    
    output_png = "individual_aug_visualization.png"
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    visualize_individual_augs()
