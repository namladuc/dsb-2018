import torch

class CFG:
    seed          = 42
    debug         = 0 
    
    # Resume Train 
    resume_train  = False
    id_wandb      = ""
    checkP_name   = ""
    epochs_res    = 0
    using_wandb   = 1
    best_dice     = -1
    best_epoch    = -1

    # ----- Dataset -----
    path_data     = "./Data"
    dataset       = "DSB2018" # 
    aug           = "kit1_3d"
    lower_percentile = 1       # Determine the lower and upper percentiles
    upper_percentile = 99
    isPinMemory   = torch.cuda.is_available()
    numWorker     = 16
    train_bs      = 8
    valid_bs      = 8
    img_size      = (384, 320) # width, height # 2D: 384x320 / 3D: 400x320 / Transformer: 384x384
    patch_size    = 160        # 3D Only / Transform = 8
    num_slice     = 5
    stride        = 1
    n_fold        = 5
    fold_selected = 1
    fold_test     = 2
    num_classes   = 3
    
    # ----- Model
    net_structure = 'Unet25D_Deeply'
    model_name    = 'Unet25D_Deeply-Size-M'
    isDeeply      = False
    backbone      = "none"
    epochs        = 50
    lr            = 0.002
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-5
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    s_channel     = 8
    use_parallel  = False
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    bilinear      = False
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_epochs  = 20
    
    # Fire Module v2 Config
    fire_split    = 2
    expand_ratio  = 4
    expand_kernel = 5
    
    # transformer config - 2
    block_num     = 4
    patch_dim     = 16
    head_num      = 4
    mlp_dim       = 512