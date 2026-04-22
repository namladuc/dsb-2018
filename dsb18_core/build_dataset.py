import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
import platform
from PIL import Image

# Sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

# PyTorch
from torch.utils.data import DataLoader

# Albumentations for augmentations
import albumentations as A

# import dataset
from .dataset.dataset2d import Dataset2D, Dataset2DTest
from .dataset.aug import get_aug_dict
from .fingerprint_utils import get_dsb2018_fingerprint


def get_dataset_mapping(CFG):
    """
    Route to DSB-2018 dataset builder (only dataset supported)
    """
    if "Unet2D" not in CFG.net_structure:
        raise ValueError(
            f"Network structure '{CFG.net_structure}' not supported. "
            f"This codebase is configured for DSB-2018 dataset only. "
            f"Use network structures like 'Unet2D_DSB2018'."
        )

    print(f"\n{'='*80}")
    print(f"Loading DSB-2018 Dataset (Network: {CFG.net_structure})")
    print(f"Configuration: {CFG.img_size}, {CFG.normalization_method} normalization")
    print(f"{'='*80}\n")
    return get_train_valid_dataset_dsb2018(CFG, CFG.path_data)


def get_train_valid_dataset_dsb2018(CFG, path_data):
    """
    Build DSB-2018 dataset with fingerprinting-based preprocessing

    Args:
        CFG: Config class to config the dataset metadata
        path_data: Path to DSB-2018 dataset folder

    Returns:
        train_loader, valid_loader: PyTorch DataLoaders
    """
    # Step 1: Get dataset fingerprint
    print("\n" + "=" * 80)
    print("STEP 1: Dataset Fingerprinting for DSB-2018")
    print("=" * 80)

    fingerprint = get_dsb2018_fingerprint(
        path_data, max_samples=None if not CFG.debug else 20, verbose=True
    )

    # Build preprocessing_params using DSB-2018 configuration
    # Use configured normalization method
    normalization_method = getattr(CFG, "normalization_method", "zscore")
    resize_mode = getattr(CFG, "resize_mode", "resize_only")

    preprocessing_params = {
        "crop_background": True,
        "crop_threshold": 0,
        "crop_margin": 10,
        "target_size": CFG.img_size,
        "resize_method": "bicubic",  # Cubic interpolation (Order 3) for images
        "mask_interpolation": "linear",  # Linear interpolation (Order 1) for masks
        "resize_mode": resize_mode,  # 'resize_only' or 'pad_and_resize'
        "intensity_normalization": {
            "method": normalization_method,
        },
    }

    # Standardize normalization method name
    normalization_method = normalization_method.replace("_", "").lower()
    preprocessing_params["intensity_normalization"]["method"] = normalization_method

    # Add normalization parameters based on method
    if normalization_method == "zscore":
        # Z-score normalization: use global mean and std from fingerprint
        preprocessing_params["intensity_normalization"]["zscore"] = {
            "mean": fingerprint["intensity"]["mean"]["mean"],
            "std": max(fingerprint["intensity"]["mean"]["std"], 1e-8),
        }
    elif normalization_method == "percentile":
        preprocessing_params["intensity_normalization"]["percentile"] = {
            "lower": getattr(CFG, "lower_percentile", 1.0),
            "upper": getattr(CFG, "upper_percentile", 99.0),
        }
    elif normalization_method == "minmax":
        preprocessing_params["intensity_normalization"]["minmax"] = {
            "min": fingerprint["intensity"]["global_min"],
            "max": fingerprint["intensity"]["global_max"],
        }

    print(f"\nDSB-2018 Preprocessing Configuration:")
    print(f"  Normalized size: {preprocessing_params['target_size']}")
    print(f"  Spacing: {CFG.spacing}")
    print(f"  Normalization method: {normalization_method}")
    print(f"  Normalization scope: {CFG.normalization_scope}")
    print(f"  Image interpolation: Order {CFG.image_interpolation} (Cubic)")
    print(f"  Mask interpolation: Order {CFG.mask_interpolation} (Linear)")
    print(f"  Crop background: {preprocessing_params['crop_background']}")
    if normalization_method == "zscore":
        print(
            f"  Global normalization - Mean: {fingerprint['intensity']['mean']['mean']:.2f}, "
            f"Std: {fingerprint['intensity']['mean']['std']:.2f}"
        )
    elif normalization_method == "minmax":
        print(
            f"  Min-Max normalization - Min: {fingerprint['intensity']['global_min']:.0f}, "
            f"Max: {fingerprint['intensity']['global_max']:.0f}"
        )

    # Step 2: Build dataframe with image paths
    print("\n" + "=" * 80)
    print("STEP 2: Building Dataset Index")
    print("=" * 80)

    train_path = os.path.join(path_data, "stage1_train")
    # Check for stage1_test or stage2_test_final
    test_path = os.path.join(path_data, "stage1_test")
    if not os.path.exists(test_path):
        test_path = os.path.join(path_data, "stage2_test_final")
    
    if not os.path.exists(train_path):
         # If train path not found, we might be in a test-only folder. 
         # Try to find stage1_train in parent if path_data looks specifically like a test folder
         parent_path = os.path.dirname(os.path.normpath(path_data))
         if os.path.exists(os.path.join(parent_path, "stage1_train")):
             train_path = os.path.join(parent_path, "stage1_train")
             print(f"  Warning: stage1_train not found in {path_data}, using {train_path} for fingerprinting")

    image_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    image_id_tests = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    if CFG.debug:
        image_ids = image_ids[:20]  # Use 20 total for train/valid split

    # Create dataframe
    data_list = []
    for image_id in image_ids:
        image_file = os.path.join(train_path, image_id, "images", f"{image_id}.png")
        masks_dir = os.path.join(train_path, image_id, "masks")

        # Get image dimensions
        img = np.array(Image.open(image_file))
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width = img.shape[:2]

        # Count masks
        mask_files = glob(os.path.join(masks_dir, "*.png"))
        num_masks = len(mask_files)

        data_list.append(
            {
                "image_id": image_id,
                "image_path": image_file,
                "masks_dir": masks_dir,
                "height": height,
                "width": width,
                "num_nuclei": num_masks,
            }
        )

    # Create dataframe test
    data_list_test = []
    for image_id in image_id_tests:
        image_file = os.path.join(test_path, image_id, "images", f"{image_id}.png")

        # Get image dimensions
        img = np.array(Image.open(image_file))
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width = img.shape[:2]

        data_list_test.append(
            {
                "image_id": image_id,
                "image_path": image_file,
                "height": height,
                "width": width,
            }
        )
    df = pd.DataFrame(data_list)
    df_test = pd.DataFrame(data_list_test)

    print(f"Total images: {len(df)}")
    print(
        f"Image size range: {df['height'].min()}x{df['width'].min()} to {df['height'].max()}x{df['width'].max()}"
    )
    print(
        f"Nuclei count: {df['num_nuclei'].min()} - {df['num_nuclei'].max()} (mean: {df['num_nuclei'].mean():.1f})"
    )

    # Step 3: Create train/validation split
    print("\n" + "=" * 80)
    print("STEP 3: Train/Validation Split")
    print("=" * 80)

    # Stratified split by number of nuclei (binned)
    # Use fewer bins for debug mode
    num_bins = 2 if CFG.debug else 5
    df["nuclei_bin"] = pd.cut(df["num_nuclei"], bins=num_bins, labels=False)

    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=CFG.seed, stratify=df["nuclei_bin"]
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # In debug mode, use limited samples
    if CFG.debug:
        train_df = train_df.iloc[:10]
        valid_df = valid_df.iloc[:2]

    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(valid_df)}")

    # Step 4: Create datasets with preprocessing
    print("\n" + "=" * 80)
    print("STEP 4: Creating Datasets with Online Preprocessing")
    print("=" * 80)

    data_transforms = get_aug_dict(CFG)

    # Store preprocessing params in CFG for dataset to use
    CFG.preprocessing_params = preprocessing_params
    CFG.dataset_type = "DSB2018"

    train_dataset = Dataset2D(train_df, CFG, subset="train", transforms=data_transforms["train"])

    valid_dataset = Dataset2D(valid_df, CFG, subset="valid", transforms=data_transforms["valid"])

    test_dataset = Dataset2DTest(df_test, CFG, transforms=data_transforms["valid"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_bs,
        num_workers=CFG.numWorker,
        shuffle=True,
        pin_memory=CFG.isPinMemory,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_bs,
        num_workers=CFG.numWorker,
        shuffle=False,
        pin_memory=CFG.isPinMemory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_bs,
        num_workers=CFG.numWorker,
        shuffle=False,
        pin_memory=CFG.isPinMemory,
    )

    print(f"\nDatasets created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("=" * 80 + "\n")

    return train_loader, valid_loader, test_loader
