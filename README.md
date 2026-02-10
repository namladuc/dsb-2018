# DSB-ImageSegmentationModel
DSB 2018 with Unet Model


Training debug code:
```sh
python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --isDeeply True --net_structure FusionUnet2D

python train.py --debug 0 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 0 --train_bs 2 --valid_bs 2 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D

python train.py --debug 1 --train_bs 1 --valid_bs 1 --numWorker 1 --using_wandb 0 --net_structure FusionUnet2D --isDeeply True

```

unet++
attention-unet
