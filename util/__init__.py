from .loss import criterion, dice_coef, iou_coef
from .util import fetch_scheduler, set_seed
from .train_valid2d import run_training2d
from .train_valid3d import run_training3d
from .train_valid2decoder import run_training2decoder

def getTrain_Valid(CFG):
    if 'Unet25D' in CFG.net_structure or 'Unet2DecoderPretrain' == CFG.net_structure:
        return run_training2d
    if 'Unet3D' in CFG.net_structure:
        return run_training3d
    if 'Unet2Decoder' == CFG.net_structure:
        return run_training2decoder
    return None