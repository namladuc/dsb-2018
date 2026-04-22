# DSB 2018 Nuclei Segmentation

PyTorch codebase for 2D nuclei segmentation on the Data Science Bowl 2018 dataset.

This project provides:
- Multiple U-Net style models (`Unet2D`, `nnUnet2D`, `FusionUnet2D`, `FusionUnet2DPlusPlus`)
- Online preprocessing with dataset fingerprinting
- Training, validation, inference, and Kaggle-style submission export
- Optional Weights and Biases logging
- ONNX export for deployment workflows

## 1. Project Overview

The repository focuses on binary nuclei segmentation from microscopy images. The training pipeline builds train/validation/test loaders from DSB 2018 folder structure, trains selected models, tracks segmentation metrics (Dice, IoU), and exports run-length encoded submission CSV files.

Main entry points:
- `train.py`: training + validation + test inference + submission generation
- `inference.py`: inference-only from a saved checkpoint
- `export2onnx.py`: export trained model checkpoint to ONNX

## 2. Problem and Dataset

Dataset layout expected by the code:

```text
data/
	dsb-data-2018/
		stage1_train/
			<image_id>/
				images/<image_id>.png
				masks/*.png
		stage2_test_final/
			<image_id>/
				images/<image_id>.png
```

How to get data:
1. Put the dataset in `data/dsb-data-2018` manually, or
2. Use `data/download_data.py` (Google Drive folder download + unzip).

Default path from configuration:
- `./data/dsb-data-2018`

## 3. Method and Model Variants

Model selection is controlled by `--net_structure`.

Supported values:
- `Unet2D`
- `nnUnet2D`
- `FusionUnet2D`
- `FusionUnet2DPlusPlus`

Notes:
- Fusion models support encoder backbones via `segmentation-models-pytorch`.
- Deep supervision can be enabled with `--isDeeply True`.

Preprocessing pipeline includes:
- Dataset fingerprinting (`dsb18_core/fingerprint_utils.py`)
- Optional crop background
- Resize / pad-and-resize
- Intensity normalization (`zscore`, `percentile`, `minmax`)

## 4. Training and Evaluation Protocol

Training script performs:
1. Seed setup for reproducibility
2. Dataset fingerprint + split + DataLoader build
3. Model build and optional checkpoint resume
4. Epoch loop with training and validation
5. Best and last checkpoint save
6. Test set inference and submission CSV export

Validation metrics:
- `dice`
- `iou`

Available losses:
- `dice_entropy`
- `tversky_entropy`
- `focal_dice`
- `bce`
- `dice`
- `tversky`

## 5. Reproducibility and Environment

### 5.1 Requirements

Use Python 3.10+ (the project formatting config targets Python 3.12).

Install dependencies:

GPU environment:
```bash
pip install -r scripts/requirements.txt
```

CPU-only environment:
```bash
pip install -r scripts/requirements-cpu.txt
```

### 5.2 Recommended baseline training command

```bash
python train.py \
	--path_data ./data/dsb-data-2018 \
	--net_structure nnUnet2D \
	--model_name nnUnet2D_DSB2018 \
	--train_bs 8 \
	--valid_bs 8 \
	--numWorker 4 \
	--epochs 50 \
	--loss_name dice_entropy \
	--metrics dice iou \
	--using_wandb 0
```

### 5.3 Fusion model example

```bash
python train.py \
	--path_data ./data/dsb-data-2018 \
	--net_structure FusionUnet2D \
	--encoder_backbone resnet18 \
	--encoder_weights imagenet \
	--s_channel 24 \
	--isDeeply True \
	--model_name FusionUnet2D_baseline
```

### 5.4 Inference from checkpoint

```bash
python inference.py \
	--path_data ./data/dsb-data-2018 \
	--net_structure FusionUnet2D \
	--encoder_backbone resnet18 \
	--encoder_weights imagenet \
	--isDeeply True \
	--checkpoint_path runs/<experiment>/<timestamp>/best_epoch_FusionUnet2D_baseline_00.bin
```

### 5.5 ONNX export

```bash
python export2onnx.py \
	--net_structure FusionUnet2D \
	--encoder_backbone resnet18 \
	--encoder_weights imagenet \
	--isDeeply True \
	--checkpoint_path runs/<experiment>/<timestamp>/best_epoch_FusionUnet2D_baseline_00.bin
```

Default ONNX output file:
- `segmentation_model_timm.onnx`

### 5.6 Weights and Biases

To enable logging:
1. Set `WANDB_API_KEY` environment variable (or edit debug login flow in `train.py`).
2. Run training with `--using_wandb 1`.

## 6. Outputs and Artifacts

Training artifacts:
- `runs/<dataset>_<net>_<encoder>_<model_name>_<aug>/<timestamp>/best_epoch_<model_name>_<fold>.bin`
- `runs/<dataset>_<net>_<encoder>_<model_name>_<aug>/<timestamp>/last_epoch_<model_name>_<fold>.bin`

Inference/export artifacts:
- `submission_<fold>.csv` (RLE-encoded masks for test images)
- `segmentation_model_timm.onnx`

## 7. Repository Structure

```text
.
|- config.py                     # Global defaults for training/inference
|- train.py                      # Main train + validate + infer pipeline
|- inference.py                  # Inference-only entrypoint
|- export2onnx.py                # Checkpoint to ONNX export
|- dsb18_core/
|  |- build_dataset.py           # Dataset indexing, split, and dataloaders
|  |- loss.py                    # Loss registry
|  |- metrics.py                 # Metric registry
|  |- pipeline/runner_2d.py      # Train/valid/inference loops
|  |- model/                     # Model implementations
|- scripts/
|  |- requirements.txt
|  |- requirements-cpu.txt
|- tests/
|  |- test_build_dataset_steps.py
|- viewer/                       # ONNX runtime viewer utilities
```

## 8. Testing

Run tests with:

```bash
pytest -q
```

Current test focus:
- Dataset fingerprinting
- Dataset build pipeline steps
- Debug preprocessing image output

## 9. Known Limitations

- This repository is focused on 2D DSB 2018-style segmentation workflows.
- Several CLI arguments use `type=bool`; pass explicit values carefully (`True`/`False`).
- Some helper scripts and subfolders are minimal/experimental and may require local adjustments.

## 10. Citation and License

License file is provided in `LICENSE`.

If you use this repository in research, cite:
- Data Science Bowl 2018 dataset/source
- Any encoder backbones or external libraries used in your experiment setup

# 11. DSB-ImageSegmentationModel
DSB 2018 with Unet Model

Training debug code:
```sh
python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --isDeeply True --net_structure FusionUnet2D

python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure Unet2D --epochs 1

python train.py --debug 0 --train_bs 2 --valid_bs 2 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D --isDeeply True

python inference.py --train_bs 1 --valid_bs 1 --numWorker 2 --using_wandb 1 --s_channel 24 --net_structure FusionUnet2D --model_name FusionUnet2D-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --isDeeply True --encoder_backbone timm-efficientnet-b3 --encoder_weights imagenet --checkpoint_path checkpoint/best_epoch_FusionUnet2D-test02_00.bin

# 22 - 03 - 2026
python train.py --train_bs 16 --valid_bs 16 --numWorker 2 --using_wandb 1 --s_channel 24 --path_data /kaggle/working/dsb-data-2018 --net_structure FusionUnet2DPlusPlus --model_name FusionUnet2DPlusPlus-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --encoder_backbone resnet18 --encoder_weights imagenet

python /kaggle/working/dsb-2018/train.py --train_bs 16 --valid_bs 16 --numWorker 2 --using_wandb 1 --s_channel 24 --path_data /kaggle/working/dsb-data-2018 --net_structure FusionUnet2DPlusPlus --model_name FusionUnet2DPlusPlus-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --encoder_backbone timm-efficientnet-b3 --encoder_weights imagenet

python /kaggle/working/dsb-2018/train.py --train_bs 16 --valid_bs 16 --numWorker 2 --using_wandb 1 --s_channel 24 --path_data /kaggle/working/dsb-data-2018 --net_structure FusionUnet2DPlusPlus --model_name FusionUnet2DPlusPlus-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --isDeeply True --encoder_backbone resnet18 --encoder_weights imagenet

python /kaggle/working/dsb-2018/train.py --train_bs 16 --valid_bs 16 --numWorker 2 --using_wandb 1 --s_channel 24 --path_data /kaggle/working/dsb-data-2018 --net_structure FusionUnet2DPlusPlus --model_name FusionUnet2DPlusPlus-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --isDeeply True --encoder_backbone timm-efficientnet-b3 --encoder_weights imagenet
```

unet++
attention-unet

# Convert

```
python export2onnx.py --s_channel 24 --net_structure FusionUnet2D --model_name FusionUnet2D-test02 --lr 0.006 --min_lr 1e-6 --isDeeply True --encoder_backbone timm-efficientnet-b3 --encoder_weights imagenet --checkpoint_path data/models/best_epoch_FusionUnet2D-test02_00.bin
```

