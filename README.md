# DSB-ImageSegmentationModel
DSB 2018 with Unet Model


Training debug code:
```sh
python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --isDeeply True --net_structure FusionUnet2D

python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure Unet2D --epochs 1

python train.py --debug 0 --train_bs 2 --valid_bs 2 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D --isDeeply True

python inference.py --train_bs 1 --valid_bs 1 --numWorker 2 --using_wandb 1 --s_channel 24 --net_structure FusionUnet2D --model_name FusionUnet2D-test02 --aug baseline --lr 0.006 --min_lr 1e-6 --normalization_method min_max --isDeeply True --encoder_backbone timm-efficientnet-b3 --encoder_weights imagenet --checkpoint_path checkpoint/best_epoch_FusionUnet2D-test02_00.bin
```

unet++
attention-unet
