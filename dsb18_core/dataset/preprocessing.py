"""
Preprocessing utilities for dataset preparation
Includes: cropping, resampling, intensity normalization, axis permutation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def crop_background(
    image: np.ndarray, mask: Optional[np.ndarray] = None, threshold: float = 0, margin: int = 10
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Crop unnecessary background/empty space from image

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Optional mask to crop along with image
        threshold: Pixels <= threshold are considered background
        margin: Pixels to add around the detected region

    Returns:
        cropped_image, cropped_mask, crop_info
    """
    # Find non-background pixels
    if len(image.shape) == 3:
        # For multi-channel, use mean across channels
        gray = image.mean(axis=2)
    else:
        gray = image

    # Find bounding box
    coords = np.argwhere(gray > threshold)

    if len(coords) == 0:
        # No foreground found, return original
        crop_info = {"cropped": False, "bbox": None}
        return image, mask, crop_info

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add margin
    h, w = gray.shape
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin + 1)

    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max] if mask is not None else None

    crop_info = {
        "cropped": True,
        "bbox": (y_min, y_max, x_min, x_max),
        "original_shape": image.shape[:2],
        "cropped_shape": cropped_image.shape[:2],
    }

    return cropped_image, cropped_mask, crop_info


def resample_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    method: str = "bilinear",
    mask: Optional[np.ndarray] = None,
    mask_interpolation: str = "linear",
    resize_mode: str = "resize_only",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Resample (resize) image to target size

    Args:
        image: Input image (H, W) or (H, W, C)
        target_size: Target size as (height, width)
        method: Interpolation method for image ('nearest', 'bilinear', 'bicubic', 'area')
        mask: Optional mask to resample along with image
        mask_interpolation: Interpolation method for mask ('nearest', 'linear', 'bicubic')
        resize_mode: 'resize_only' (direct resize) or 'pad_and_resize' (preserve aspect ratio with padding)

    Returns:
        resampled_image, resampled_mask
    """
    target_w, target_h = target_size

    # Map method names to cv2 constants
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
        "linear": cv2.INTER_LINEAR,  # Alias for bilinear
    }

    interp = interpolation_map.get(method, cv2.INTER_LINEAR)
    mask_interp = interpolation_map.get(mask_interpolation, cv2.INTER_LINEAR)

    # Handle different resize modes
    if resize_mode == "pad_and_resize":
        # Preserve aspect ratio with padding
        h, w = image.shape[:2]
        aspect_ratio = w / h
        target_aspect = target_w / target_h

        if aspect_ratio > target_aspect:
            # Width is relatively larger - pad height
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Height is relatively larger - pad width
            new_h = target_h
            new_w = int(target_h * aspect_ratio)

        # Resize to intermediate size
        resampled_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        if mask is not None:
            resampled_mask = cv2.resize(mask, (new_w, new_h), interpolation=mask_interp)

        # Pad to target size (center padding)
        pad_h_top = (target_h - new_h) // 2
        pad_h_bottom = target_h - new_h - pad_h_top
        pad_w_left = (target_w - new_w) // 2
        pad_w_right = target_w - new_w - pad_w_left

        # Determine padding color (0 for black)
        if len(resampled_image.shape) == 3:
            pad_color = (0, 0, 0)
        else:
            pad_color = 0

        resampled_image = cv2.copyMakeBorder(
            resampled_image,
            pad_h_top,
            pad_h_bottom,
            pad_w_left,
            pad_w_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        if mask is not None:
            # Pad mask with zeros
            resampled_mask = cv2.copyMakeBorder(
                resampled_mask,
                pad_h_top,
                pad_h_bottom,
                pad_w_left,
                pad_w_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
    else:
        # Direct resize (resize_only mode)
        resampled_image = cv2.resize(image, (target_w, target_h), interpolation=interp)

    # Resize mask (only if not already done in pad_and_resize mode)
    if resize_mode == "resize_only" and mask is not None:
        # For masks with linear interpolation, we may get values between 0 and 1
        # This is intentional for smooth resampling; they'll be thresholded later if needed
        resampled_mask = cv2.resize(mask, (target_w, target_h), interpolation=mask_interp)
    elif resize_mode == "resize_only":
        resampled_mask = None
    else:
        # pad_and_resize mode - mask already resampled above
        resampled_mask = resampled_mask if mask is not None else None

    return resampled_image, resampled_mask


def normalize_intensity(
    image: np.ndarray, method: str = "min_max", params: Optional[Dict] = None
) -> np.ndarray:
    """
    Normalize intensity values

    Args:
        image: Input image
        method: Normalization method ('min_max', 'z_score', 'percentile')
        params: Parameters for normalization method

    Returns:
        normalized_image
    """
    image = image.astype(np.float32)

    if params is None:
        params = {}

    if method == "min_max":
        # Scale to [0, 1]
        min_val = params.get("min", image.min())
        max_val = params.get("max", image.max())

        if max_val - min_val > 1e-8:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = image - min_val

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)

    elif method == "z_score":
        # Standardize: (x - mean) / std
        mean = params.get("mean", image.mean())
        std = params.get("std", image.std())

        if std > 1e-8:
            normalized = (image - mean) / std
        else:
            normalized = image - mean

    elif method == "percentile":
        # Clip using percentiles then scale to [0, 1]
        lower_p = params.get("lower", 1.0)
        upper_p = params.get("upper", 99.0)

        lower_val = np.percentile(image, lower_p)
        upper_val = np.percentile(image, upper_p)

        # Clip
        clipped = np.clip(image, lower_val, upper_val)

        # Scale to [0, 1]
        if upper_val - lower_val > 1e-8:
            normalized = (clipped - lower_val) / (upper_val - lower_val)
        else:
            normalized = clipped - lower_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def permute_axes(
    image: np.ndarray, source_order: str = "HWC", target_order: str = "CHW"
) -> np.ndarray:
    """
    Permute image axes

    Args:
        image: Input image
        source_order: Source axis order (e.g., 'HWC', 'WHC')
        target_order: Target axis order (e.g., 'CHW', 'HWC')

    Returns:
        permuted_image
    """
    if source_order == target_order:
        return image

    # Map axis labels to indices
    axis_map = {"H": 0, "W": 1, "C": 2}

    # Get current and target axis positions
    source_indices = [axis_map[c] for c in source_order]
    target_indices = [axis_map[c] for c in target_order]

    # Create permutation
    perm = [source_indices.index(i) for i in target_indices]

    return np.transpose(image, perm)


def preprocess_pipeline(
    image: np.ndarray, mask: Optional[np.ndarray] = None, config: Optional[Dict] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Complete preprocessing pipeline

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Optional mask
        config: Preprocessing configuration dictionary

    Returns:
        processed_image, processed_mask, preprocessing_info
    """
    if config is None:
        config = {}

    preprocessing_info = {}

    # Step 1: Crop background
    if config.get("crop_background", False):
        image, mask, crop_info = crop_background(
            image,
            mask,
            threshold=config.get("crop_threshold", 0),
            margin=config.get("crop_margin", 10),
        )
        preprocessing_info["crop"] = crop_info

    # Step 2: Resample to target size
    target_size = config.get("target_size", None)
    if target_size is not None:
        original_shape = image.shape[:2]
        image, mask = resample_image(
            image,
            target_size,
            method=config.get("resize_method", "bilinear"),
            mask=mask,
            mask_interpolation=config.get("mask_interpolation", "linear"),
            resize_mode=config.get("resize_mode", "resize_only"),
        )
        preprocessing_info["resample"] = {
            "original_size": original_shape,
            "target_size": target_size,
            "method": config.get("resize_method", "bilinear"),
            "mask_interpolation": config.get("mask_interpolation", "linear"),
        }

    # Step 3: Intensity normalization
    norm_method = config.get("intensity_normalization", {}).get("method", "min_max")
    norm_params = config.get("intensity_normalization", {}).get(norm_method, {})

    image = normalize_intensity(image, method=norm_method, params=norm_params)
    preprocessing_info["normalization"] = {"method": norm_method, "params": norm_params}

    # Ensure 3D for axis permutation
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if mask is not None and len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]

    # Step 4: Axis permutation (if needed, but usually done by transforms)
    # This is typically handled by PyTorch transforms, so we skip it here
    # The transpose will be done in Dataset.__getitem__

    preprocessing_info["final_shape"] = image.shape
    preprocessing_info["dtype"] = str(image.dtype)

    return image, mask, preprocessing_info


def batch_preprocess(
    images: list, masks: Optional[list] = None, config: Optional[Dict] = None
) -> Tuple[list, Optional[list], list]:
    """
    Preprocess a batch of images

    Args:
        images: List of images
        masks: Optional list of masks
        config: Preprocessing configuration

    Returns:
        processed_images, processed_masks, preprocessing_infos
    """
    processed_images = []
    processed_masks = [] if masks is not None else None
    preprocessing_infos = []

    for i, image in enumerate(images):
        mask = masks[i] if masks is not None else None

        proc_img, proc_mask, info = preprocess_pipeline(image, mask, config)

        processed_images.append(proc_img)
        if processed_masks is not None:
            processed_masks.append(proc_mask)
        preprocessing_infos.append(info)

    return processed_images, processed_masks, preprocessing_infos
