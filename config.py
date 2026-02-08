import torch

class CFG:
    seed          = 42
    debug         = 0  # Debug mode: saves first 5 images before/after preprocessing to debug/ folder 
    
    # Resume Train 
    resume_train  = False
    id_wandb      = ""
    checkP_name   = ""
    epochs_res    = 0
    using_wandb   = 0
    best_dice     = -1
    best_epoch    = -1

    # ----- Dataset -----
    path_data     = "./data"
    dataset       = "DSB2018" # DSB-2018 Nuclei Segmentation
    aug           = "baseline"
    isPinMemory   = torch.cuda.is_available()
    numWorker     = 16
    train_bs      = 8
    valid_bs      = 8
    img_size      = (320, 256) # width, height
    resize_mode   = 'pad_and_resize' # 'resize_only' or 'pad_and_resize'
    
    # ----- Preprocessing Config -----
    spacing = (1, 1)                 # x, y spacing
    normalization_method = 'z_score'  # 'z_score' or 'percentile' or 'minmax'
    normalization_scope = 'global'   # apply normalization across entire image
    image_interpolation = 3          # Order 3 = Cubic Interpolation
    mask_interpolation = 1           # Order 1 = Linear Interpolation
    
    # ----- Model -----
    net_structure = 'Unet2D_DSB2018'
    model_name    = 'Unet2D_DSB2018'
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