import cv2
import numpy as np
import torch
from PIL import Image
from glob import glob
import os
from .preprocessing import preprocess_pipeline


class Dataset2D(torch.utils.data.Dataset):
    def __init__(self, df, CFG, subset="train", transforms=None):
        """[Dataset2D]
        Args:
            df (_type_): _DataFrame preprocessing for DSB-2018 dataset metadata _
            CFG (_type_): _Config class_
            subset (str, optional): _'train' / 'valid' / 'test'_. Defaults to "train".
            transforms (_type_, optional): _Augmentation_. Defaults to None.
        """
        self.df = df
        self.subset = subset
        self.transforms = transforms

        # Common parameters
        self.width_norm = CFG.img_size[0] if hasattr(CFG, "img_size") else 256
        self.height_norm = CFG.img_size[1] if hasattr(CFG, "img_size") else 256

        # DSB-2018 preprocessing parameters
        self.preprocessing_params = getattr(CFG, "preprocessing_params", None)
        if self.preprocessing_params is None:
            # Default preprocessing for DSB-2018 with zscore normalization
            self.preprocessing_params = {
                "crop_background": True,
                "crop_threshold": 0,
                "crop_margin": 10,
                "target_size": [self.width_norm, self.height_norm],
                "resize_method": "bicubic",  # Cubic interpolation for images (Order 3)
                "resize_mode": getattr(
                    CFG, "resize_mode", "resize_only"
                ),  # 'resize_only' or 'pad_and_resize'
                "mask_interpolation": "linear",
                "intensity_normalization": {
                    "method": "zscore",  # Z-score normalization for DSB-2018
                    "zscore": {"mean": 0.0, "std": 1.0},  # Will be computed from fingerprint
                },
            }

        # Store CFG for access to interpolation settings
        self.CFG = CFG

        # Setup debug directory if debug mode is enabled
        self.debug_mode = getattr(CFG, "debug", False) and CFG.debug
        if self.debug_mode:
            self.debug_dir = os.path.join(os.getcwd(), "debug", subset)
            os.makedirs(os.path.join(self.debug_dir, "before_preprocess"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "after_preprocess"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "masks_before"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "masks_after"), exist_ok=True)
            print(
                f"  [Dataset2D] Debug mode enabled for {subset} subset - saving first 5 samples to {self.debug_dir}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load and preprocess image and mask for DSB-2018"""
        # Check debug mode dynamically from CFG
        debug_flag = getattr(self.CFG, "debug", False)
        debug_mode = bool(debug_flag)  # Convert to boolean (handles 0, 1, True, False)

        if debug_mode and index < 5:
            print(
                f"[DEBUG] Dataset2D.__getitem__ called: index={index}, subset={self.subset}, debug_mode={debug_mode}",
                flush=True,
            )

        # Get paths
        image_path = self.df["image_path"].iloc[index]
        masks_dir = self.df["masks_dir"].iloc[index]

        # Load image
        img = np.array(Image.open(image_path))

        # Ensure 3-channel (convert grayscale to RGB if needed)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        # Load and combine all masks
        h, w = img.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)

        if self.subset == "train" or self.subset == "valid":
            mask_files = glob(f"{masks_dir}/*.png")
            for mask_file in mask_files:
                mask = np.array(Image.open(mask_file))
                # Binary mask for each nucleus
                combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.float32))

        # Assert image and label have the same size before preprocessing
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = combined_mask.shape[:2]
        assert img_h == mask_h and img_w == mask_w, (
            f"Image shape {(img_h, img_w)} does not match mask shape {(mask_h, mask_w)} "
            f"for image_id {self.df['image_id'].iloc[index]} at path {image_path}"
        )

        # Save before preprocessing (debug mode)
        if debug_mode and index < 5:  # Only save first 5 samples per subset
            print(
                f"[DEBUG] Saving debug images for index={index}, image_id={self.df['image_id'].iloc[index]}",
                flush=True,
            )
            image_id = self.df["image_id"].iloc[index]
            # Save original image
            img_before = img.copy()
            if img_before.dtype != np.uint8:
                img_before = (
                    (img_before * 255).astype(np.uint8)
                    if img_before.max() <= 1.0
                    else img_before.astype(np.uint8)
                )

            # Ensure debug directory exists
            debug_dir = os.path.join(os.getcwd(), "debug", self.subset)
            os.makedirs(os.path.join(debug_dir, "before_preprocess"), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, "masks_before"), exist_ok=True)

            save_path = os.path.join(debug_dir, "before_preprocess", f"{image_id}_idx{index}.png")
            print(f"[DEBUG] Saving before_preprocess to: {save_path}", flush=True)
            Image.fromarray(img_before).save(save_path)
            # Save original mask
            mask_before = (combined_mask * 255).astype(np.uint8)
            Image.fromarray(mask_before).save(
                os.path.join(debug_dir, "masks_before", f"{image_id}_idx{index}.png")
            )

        # Apply preprocessing pipeline
        if self.preprocessing_params is not None:
            img, mask_processed, _ = preprocess_pipeline(
                img, combined_mask if combined_mask.sum() > 0 else None, self.preprocessing_params
            )
            if mask_processed is not None:
                combined_mask = (
                    mask_processed.squeeze() if len(mask_processed.shape) == 3 else mask_processed
                )
            else:
                # If no mask returned, resize mask manually
                target_size = self.preprocessing_params.get("target_size", [h, w])
                combined_mask = cv2.resize(
                    combined_mask, target_size, interpolation=cv2.INTER_NEAREST
                )

        # Debug: Print pixel value statistics after preprocessing
        if debug_mode and index < 5:
            print(f"[DEBUG] Image pixel values after preprocessing:")
            print(f"  Shape: {img.shape}")
            print(f"  Min: {img.min():.4f}, Max: {img.max():.4f}")
            print(f"  Mean: {img.mean():.4f}, Std: {img.std():.4f}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Range check - values in [-5, 5]: {np.percentile(img, [0, 25, 50, 75, 100])}")
            print(f"[DEBUG] Mask pixel values after preprocessing:")
            print(f"  Shape: {combined_mask.shape}")
            print(f"  Min: {combined_mask.min():.4f}, Max: {combined_mask.max():.4f}")
            print(f"  Mean: {combined_mask.mean():.4f}")
            print()

        # Ensure correct shapes
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        if len(combined_mask.shape) == 3:
            combined_mask = combined_mask[:, :, 0]

        # Add mask as third dimension if needed (for compatibility)
        masks = combined_mask[:, :, np.newaxis].astype(np.float32)

        # Save after preprocessing (debug mode)
        if debug_mode and index < 5:  # Only save first 5 samples per subset
            image_id = self.df["image_id"].iloc[index]
            # Save preprocessed image (denormalize for visualization)
            img_after = img.copy()
            # Normalize to 0-255 range for saving
            img_min, img_max = img_after.min(), img_after.max()
            if img_max > img_min:
                img_after = ((img_after - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_after = np.zeros_like(img_after, dtype=np.uint8)
            if img_after.shape[2] == 1:
                img_after = np.repeat(img_after, 3, axis=2)

            # Ensure debug directory exists
            debug_dir = os.path.join(os.getcwd(), "debug", self.subset)
            os.makedirs(os.path.join(debug_dir, "after_preprocess"), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, "masks_after"), exist_ok=True)

            Image.fromarray(img_after).save(
                os.path.join(debug_dir, "after_preprocess", f"{image_id}_idx{index}.png")
            )
            # Save preprocessed mask
            mask_after = (masks[:, :, 0] * 255).astype(np.uint8)
            Image.fromarray(mask_after).save(
                os.path.join(debug_dir, "masks_after", f"{image_id}_idx{index}.png")
            )

        # Apply augmentations
        if self.transforms:
            data = self.transforms(image=img, mask=masks)
            img = data["image"]
            masks = data["mask"]

        # Transpose to CHW format (required by PyTorch)
        img = img.transpose(2, 0, 1)
        masks = masks.transpose(2, 0, 1)

        if self.subset == "train" or self.subset == "valid":
            return torch.tensor(img, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)
        else:
            return torch.tensor(img, dtype=torch.float32), self.df["image_id"].iloc[index], h, w


class Dataset2DTest(torch.utils.data.Dataset):
    def __init__(self, df, CFG, transforms=None):
        """Dataset2D for test dataset without masks

        Args:
            df (_type_): _DataFrame preprocessing for DSB-2018 dataset metadata _
            CFG (_type_): _Config class_
            transforms (_type_, optional): _Augmentation_. Defaults to None.
        """
        self.df = df
        self.transforms = transforms

        # Common parameters
        self.width_norm = CFG.img_size[0] if hasattr(CFG, "img_size") else 256
        self.height_norm = CFG.img_size[1] if hasattr(CFG, "img_size") else 256

        # DSB-2018 preprocessing parameters
        self.preprocessing_params = getattr(CFG, "preprocessing_params", None)
        if self.preprocessing_params is None:
            # Default preprocessing for DSB-2018 with zscore normalization
            self.preprocessing_params = {
                "crop_background": True,
                "crop_threshold": 0,
                "crop_margin": 10,
                "target_size": [self.width_norm, self.height_norm],
                "resize_method": "bicubic",  # Cubic interpolation for images (Order 3)
                "resize_mode": getattr(
                    CFG, "resize_mode", "resize_only"
                ),  # 'resize_only' or 'pad_and_resize'
                "intensity_normalization": {
                    "method": "zscore",  # Z-score normalization for DSB-2018
                    "zscore": {"mean": 0.0, "std": 1.0},  # Will be computed from fingerprint
                },
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load and preprocess image for DSB-2018 test dataset without masks"""
        # Get paths
        image_path = self.df["image_path"].iloc[index]

        # Load image
        img = np.array(Image.open(image_path))

        # Ensure 3-channel (convert grayscale to RGB if needed)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        # Apply preprocessing pipeline
        if self.preprocessing_params is not None:
            img, _, meta_normalization = preprocess_pipeline(img, None, self.preprocessing_params)

        # Apply augmentations
        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]

        # Transpose to CHW format (required by PyTorch)
        img = img.transpose(2, 0, 1)

        return (
            torch.tensor(img, dtype=torch.float32),
            self.df["image_id"].iloc[index],
            meta_normalization,
        )
