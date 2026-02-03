# Model
import torch
from .Model.unet2d import UNet


# Dataset
from .build_dataset import getDatasetMapping

def getModel(CFG):
    model_mapping = {
        # ------------------------------------------------ 
        # ------------------------------------------------ 
        # ----------- 2D MODEL BASELINE ------------------ 
        # ------------------------------------------------ 
        # ------------------------------------------------
        "Unet25D": UNet(
            in_channels=CFG.num_slice, 
            n_channels=CFG.s_channel, 
            n_classes=CFG.num_classes),
    }
    
    if CFG.net_structure not in model_mapping.keys():
        raise ValueError("Network " + CFG.net_structure + " unknown!")
    
    model = model_mapping[CFG.net_structure]
    
    if CFG.use_parallel:
        model = torch.nn.DataParallel(model)
    
    if CFG.debug:
        print("Params: ", sum(p.numel() for p in model.parameters()))

    return model