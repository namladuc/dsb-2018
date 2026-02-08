from .loss import criterion, dice_coef, iou_coef
from .utils import fetch_scheduler, set_seed
from .train_valid2d import run_training2d

def get_train_valid(CFG):
    if 'Unet25D' in CFG.net_structure or 'Unet2DecoderPretrain' == CFG.net_structure or 'Unet2D' in CFG.net_structure:
        return run_training2d
    raise ValueError(f"Network structure '{CFG.net_structure}' not supported. "
                    f"This codebase supports 2D models only (DSB-2018 dataset).")