"""Test suite for dataset building and preprocessing visualization"""
import os
import sys
import shutil

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CFG
from dsb18_core.build_dataset import (
    get_dataset_mapping,
    get_train_valid_dataset_dsb2018,
)
from dsb18_core.fingerprint_utils import get_dsb2018_fingerprint
from dsb18_core.utils import set_seed as _set_seed

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _data_root() -> str:
    return os.path.join(_repo_root(), "data", "dsb-data-2018")


def _debug_dir() -> str:
    return os.path.join(_repo_root(), "debug")


def _configure_cfg(debug_mode: int = 1) -> None:
    _set_seed(CFG.seed)  # Ensure seed is set when config is modified
    CFG.debug = debug_mode
    CFG.path_data = _data_root()
    CFG.net_structure = "Unet2D_DSB2018"
    CFG.numWorker = 1


def test_step_1_fingerprint() -> None:
    """Test Step 1: Dataset fingerprinting"""
    _set_seed(CFG.seed)
    print("\n" + "="*60)
    print("TEST: Step 1 - Dataset Fingerprinting")
    print("="*60)
    _configure_cfg(debug_mode=1)
    fingerprint = get_dsb2018_fingerprint(
        CFG.path_data,
        max_samples=5,
        verbose=False,
    )
    assert isinstance(fingerprint, dict)
    assert "intensity" in fingerprint
    print("✓ Fingerprinting passed")


def test_step_2_3_4_dataset_build() -> None:
    """Test Steps 2-4: Dataset index building, train/valid split, and dataset creation"""
    _set_seed(CFG.seed)
    print("\n" + "="*60)
    print("TEST: Steps 2-4 - Dataset Building")
    print("="*60)
    _configure_cfg(debug_mode=1)
    train_loader, valid_loader = get_train_valid_dataset_dsb2018(
        CFG,
        CFG.path_data,
    )
    assert len(train_loader) > 0
    assert len(valid_loader) > 0

    images, masks = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert images.shape[0] > 0
    assert masks.shape[0] > 0
    print("✓ Dataset building passed")


def test_dataset_mapping_entrypoint() -> None:
    """Test the main dataset mapping entrypoint"""
    _set_seed(CFG.seed)
    print("\n" + "="*60)
    print("TEST: Dataset Mapping Entrypoint")
    print("="*60)
    _configure_cfg(debug_mode=1)
    train_loader, valid_loader = get_dataset_mapping(CFG)
    assert len(train_loader) > 0
    assert len(valid_loader) > 0
    print("✓ Dataset mapping entrypoint passed")


def test_debug_preprocessing_output() -> None:
    """Test that debug mode saves preprocessing images"""
    _set_seed(CFG.seed)
    print("\n" + "="*60)
    print("TEST: Debug Mode Preprocessing Visualization")
    print("="*60)
    
    # Enable debug mode
    _configure_cfg(debug_mode=True)
    CFG.numWorker = 0  # Single worker for deterministic behavior
    CFG.train_bs = 1   # Batch size of 1 to ensure we process indices sequentially
    CFG.valid_bs = 1
    
    print(f"  CFG.debug is set to: {CFG.debug}")
    
    # Clean up existing debug folder
    debug_path = _debug_dir()
    if os.path.exists(debug_path):
        shutil.rmtree(debug_path)
    print(f"  Cleaned up debug folder: {debug_path}")
    
    # Get dataset - this should trigger debug image saving
    train_loader, valid_loader = get_dataset_mapping(CFG)
    
    # Iterate through batches and directly load first 5 samples to ensure we get indices 0-4
    print(f"  Loading first 5 samples to trigger debug saving...", flush=True)
    train_dataset = train_loader.dataset
    
    for idx in range(min(5, len(train_dataset))):
        print(f"    Loading sample index {idx}...", flush=True)
        _ = train_dataset[idx]  # Directly call __getitem__ to ensure we get exact indices
    
    # Force flush and wait a moment for filesystem
    import sys
    sys.stdout.flush()
    import time
    time.sleep(0.5)
    
    # Check that debug directories were created
    print(f"  Checking debug directories...")
    assert os.path.exists(os.path.join(debug_path, "train")), "Debug train folder not created"
    assert os.path.exists(os.path.join(debug_path, "train", "before_preprocess")), "Before preprocess folder not created"
    assert os.path.exists(os.path.join(debug_path, "train", "after_preprocess")), "After preprocess folder not created"
    assert os.path.exists(os.path.join(debug_path, "train", "masks_before")), "Masks before folder not created"
    assert os.path.exists(os.path.join(debug_path, "train", "masks_after")), "Masks after folder not created"
    
    # Check that images were saved (should be up to 5 per subset)
    before_images = os.listdir(os.path.join(debug_path, "train", "before_preprocess"))
    after_images = os.listdir(os.path.join(debug_path, "train", "after_preprocess"))
    before_masks = os.listdir(os.path.join(debug_path, "train", "masks_before"))
    after_masks = os.listdir(os.path.join(debug_path, "train", "masks_after"))
    
    print(f"  Debug images saved:")
    print(f"    Before preprocessing: {len(before_images)} images")
    print(f"    After preprocessing: {len(after_images)} images")
    print(f"    Masks before: {len(before_masks)} masks")
    print(f"    Masks after: {len(after_masks)} masks")
    
    assert len(before_images) > 0, "No before preprocessing images saved"
    assert len(after_images) > 0, "No after preprocessing images saved"
    assert len(before_masks) > 0, "No before masks saved"
    assert len(after_masks) > 0, "No after masks saved"
    
    # Verify same number of images/masks
    assert len(before_images) == len(after_images), "Mismatch in before/after image count"
    assert len(before_masks) == len(after_masks), "Mismatch in before/after mask count"
    
    # Verify file timestamps are recent (created during this test)
    now = time.time()
    for img_file in before_images:
        img_path = os.path.join(debug_path, "train", "before_preprocess", img_file)
        file_mtime = os.path.getmtime(img_path)
        assert now - file_mtime < 10, f"File {img_file} is not fresh (older than 10 seconds)"
    
    print(f"  ✓ Debug mode successfully saved fresh preprocessing visualizations to {debug_path}")
    print(f"    Check the debug folder to see before/after preprocessing results")


if __name__ == "__main__":
    # Set global seed for reproducibility
    _set_seed(CFG.seed)
    
    print("\n" + "="*60)
    print("DATASET BUILD & DEBUG TEST SUITE")
    print("="*60)
    
    test_step_1_fingerprint()
    test_step_2_3_4_dataset_build()
    test_dataset_mapping_entrypoint()
    test_debug_preprocessing_output()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
