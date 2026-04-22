"""
Utility functions for dataset fingerprinting and preprocessing configuration
"""

import numpy as np
import pandas as pd
from glob import glob
import os
from PIL import Image
from collections import defaultdict
import json


def get_dsb2018_fingerprint(data_root, max_samples=None, verbose=True):
    """
    Analyze DSB-2018 dataset to extract fingerprint statistics

    Args:
        data_root: Path to DSB-2018 data directory
        max_samples: Maximum number of samples to analyze (None = all)
        verbose: Print progress

    Returns:
        Dictionary containing dataset statistics and recommended preprocessing params
    """
    train_path = os.path.join(data_root, "stage1_train")
    if not os.path.exists(train_path):
        # Fallback: look for stage1_train in parent if data_root points to a subfolder
        parent_path = os.path.dirname(os.path.normpath(data_root))
        if os.path.exists(os.path.join(parent_path, "stage1_train")):
            train_path = os.path.join(parent_path, "stage1_train")

    # Get all image IDs
    image_ids = []
    if os.path.exists(train_path):
        image_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    if not image_ids:
        # Final emergency fallback: try to find any folder with 'images' subfolder nearby 
        # (This is just to avoid complete failure if structure is non-standard)
        print(f"  Warning: No training images found in {train_path}. Normalization constants may be local-only.")

    if max_samples:
        image_ids = image_ids[:max_samples]

    if verbose:
        print(f"Analyzing {len(image_ids)} images from DSB-2018 training set...")

    stats = {
        "heights": [],
        "widths": [],
        "channels": [],
        "intensities_min": [],
        "intensities_max": [],
        "intensities_mean": [],
        "intensities_std": [],
        "num_nuclei": [],
        "mask_areas": [],
    }

    for idx, image_id in enumerate(image_ids):
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_ids)} images...")

        try:
            image_path = os.path.join(train_path, image_id, "images", f"{image_id}.png")
            masks_path = os.path.join(train_path, image_id, "masks")

            # Load image
            img = np.array(Image.open(image_path))

            # Image size info
            if len(img.shape) == 2:
                height, width = img.shape
                channels = 1
            else:
                height, width, channels = img.shape

            stats["heights"].append(height)
            stats["widths"].append(width)
            stats["channels"].append(channels)

            # Intensity information
            img_float = img.astype(np.float32)
            stats["intensities_min"].append(img_float.min())
            stats["intensities_max"].append(img_float.max())
            stats["intensities_mean"].append(img_float.mean())
            stats["intensities_std"].append(img_float.std())

            # Analyze masks
            if os.path.exists(masks_path):
                mask_files = glob(os.path.join(masks_path, "*.png"))
                num_nuclei = len(mask_files)
                stats["num_nuclei"].append(num_nuclei)

                for mask_file in mask_files:
                    mask = np.array(Image.open(mask_file))
                    mask_area = np.sum(mask > 0)
                    stats["mask_areas"].append(mask_area)

        except Exception as e:
            if verbose:
                print(f"Error processing {image_id}: {e}")
            continue

    # Calculate summary statistics
    fingerprint = {
        "dataset_name": "DSB-2018",
        "total_images": len(image_ids),
        "image_sizes": {
            "height": {
                "min": int(np.min(stats["heights"])),
                "max": int(np.max(stats["heights"])),
                "mean": float(np.mean(stats["heights"])),
                "median": float(np.median(stats["heights"])),
                "std": float(np.std(stats["heights"])),
                "percentile_5": float(np.percentile(stats["heights"], 5)),
                "percentile_95": float(np.percentile(stats["heights"], 95)),
            },
            "width": {
                "min": int(np.min(stats["widths"])),
                "max": int(np.max(stats["widths"])),
                "mean": float(np.mean(stats["widths"])),
                "median": float(np.median(stats["widths"])),
                "std": float(np.std(stats["widths"])),
                "percentile_5": float(np.percentile(stats["widths"], 5)),
                "percentile_95": float(np.percentile(stats["widths"], 95)),
            },
        },
        "channels": {
            "distribution": dict(pd.Series(stats["channels"]).value_counts().to_dict()),
            "most_common": int(pd.Series(stats["channels"]).mode()[0]),
        },
        "intensity": {
            "global_min": float(np.min(stats["intensities_min"])),
            "global_max": float(np.max(stats["intensities_max"])),
            "mean": {
                "mean": float(np.mean(stats["intensities_mean"])),
                "std": float(np.std(stats["intensities_mean"])),
                "percentile_1": float(np.percentile(stats["intensities_mean"], 1)),
                "percentile_99": float(np.percentile(stats["intensities_mean"], 99)),
            },
            "std": {
                "mean": float(np.mean(stats["intensities_std"])),
                "std": float(np.std(stats["intensities_std"])),
            },
        },
        "nuclei_statistics": {
            "per_image": {
                "min": int(np.min(stats["num_nuclei"])),
                "max": int(np.max(stats["num_nuclei"])),
                "mean": float(np.mean(stats["num_nuclei"])),
                "median": float(np.median(stats["num_nuclei"])),
            },
            "nucleus_area": {
                "min": float(np.min(stats["mask_areas"])) if stats["mask_areas"] else 0,
                "max": float(np.max(stats["mask_areas"])) if stats["mask_areas"] else 0,
                "mean": float(np.mean(stats["mask_areas"])) if stats["mask_areas"] else 0,
                "median": float(np.median(stats["mask_areas"])) if stats["mask_areas"] else 0,
            },
        },
    }

    # Generate recommended preprocessing parameters
    fingerprint["recommended_preprocessing"] = generate_preprocessing_params(fingerprint)

    if verbose:
        print("\n" + "=" * 80)
        print("DATASET FINGERPRINT SUMMARY")
        print("=" * 80)
        print(f"Total images: {fingerprint['total_images']}")
        print(f"\nImage sizes:")
        print(
            f"  Height: {fingerprint['image_sizes']['height']['min']} - {fingerprint['image_sizes']['height']['max']} "
            f"(mean: {fingerprint['image_sizes']['height']['mean']:.1f})"
        )
        print(
            f"  Width:  {fingerprint['image_sizes']['width']['min']} - {fingerprint['image_sizes']['width']['max']} "
            f"(mean: {fingerprint['image_sizes']['width']['mean']:.1f})"
        )
        print(
            f"\nIntensity range: [{fingerprint['intensity']['global_min']:.1f}, {fingerprint['intensity']['global_max']:.1f}]"
        )
        print(
            f"Mean intensity: {fingerprint['intensity']['mean']['mean']:.2f} ± {fingerprint['intensity']['mean']['std']:.2f}"
        )
        print(
            f"\nNuclei per image: {fingerprint['nuclei_statistics']['per_image']['mean']:.1f} "
            f"(min: {fingerprint['nuclei_statistics']['per_image']['min']}, "
            f"max: {fingerprint['nuclei_statistics']['per_image']['max']})"
        )
        print("=" * 80 + "\n")

    return fingerprint


def generate_preprocessing_params(fingerprint):
    """
    Generate recommended preprocessing parameters based on fingerprint statistics

    Args:
        fingerprint: Dictionary containing dataset statistics

    Returns:
        Dictionary with preprocessing recommendations
    """
    # Calculate target size (use median and round to nearest multiple of 32 for model compatibility)
    target_height = int(np.round(fingerprint["image_sizes"]["height"]["median"] / 32) * 32)
    target_width = int(np.round(fingerprint["image_sizes"]["width"]["median"] / 32) * 32)

    # Clamp to reasonable ranges
    target_height = max(256, min(1024, target_height))
    target_width = max(256, min(1024, target_width))

    # Calculate intensity normalization parameters
    # Use percentiles to be robust to outliers
    intensity_lower = fingerprint["intensity"]["global_min"]
    intensity_upper = fingerprint["intensity"]["global_max"]

    # For standard normalization
    intensity_mean = fingerprint["intensity"]["mean"]["mean"]
    intensity_std = fingerprint["intensity"]["mean"]["std"]

    params = {
        "target_size": [target_height, target_width],
        "original_size_range": {
            "height": [
                fingerprint["image_sizes"]["height"]["min"],
                fingerprint["image_sizes"]["height"]["max"],
            ],
            "width": [
                fingerprint["image_sizes"]["width"]["min"],
                fingerprint["image_sizes"]["width"]["max"],
            ],
        },
        "resize_method": "bilinear",  # or 'bicubic' for higher quality
        "crop_background": True,
        "crop_threshold": 0,  # Pixels with value <= threshold are considered background
        "intensity_normalization": {
            "method": "min_max",  # Options: 'min_max', 'z_score', 'percentile'
            "min_max": {"min": intensity_lower, "max": intensity_upper},
            "z_score": {
                "mean": intensity_mean,
                "std": max(intensity_std, 1e-8),  # Avoid division by zero
            },
            "percentile": {"lower": 1.0, "upper": 99.0},  # 1st percentile  # 99th percentile
        },
        "axis_order": "CHW",  # Channel, Height, Width (PyTorch convention)
        "dtype": "float32",
    }

    return params


def save_fingerprint(fingerprint, output_path):
    """Save fingerprint to JSON file"""
    with open(output_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"Fingerprint saved to: {output_path}")


def load_fingerprint(input_path):
    """Load fingerprint from JSON file"""
    with open(input_path, "r") as f:
        fingerprint = json.load(f)
    return fingerprint
