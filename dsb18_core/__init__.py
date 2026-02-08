import torch
from .model.unet2d import UNet
from .model.nn_unet2d import nnUNet2D
from .build_dataset import get_dataset_mapping


def get_model(CFG):
    """Get segmentation model by configuration.
    
    Args:
        CFG: Configuration object with net_structure and model parameters
        
    Returns:
        PyTorch model instance
        
    Raises:
        ValueError: If network structure is not supported
    """
    model_mapping = {
        "Unet2D": UNet(
            in_channels=CFG.input_channel, 
            n_channels=CFG.s_channel, 
            n_classes=CFG.num_classes,
            isDeeply=CFG.isDeeply
        ),
        "nnUnet2D": nnUNet2D(
            in_channels=CFG.input_channel, 
            n_channels=CFG.s_channel, 
            n_classes=CFG.num_classes,
            isDeeply=CFG.isDeeply
        ),
    }
    
    if CFG.net_structure not in model_mapping:
        raise ValueError(f"Network '{CFG.net_structure}' not supported. "
                        f"Available: {list(model_mapping.keys())}")
    
    model = model_mapping[CFG.net_structure]
    
    if CFG.use_parallel:
        model = torch.nn.DataParallel(model)
    
    if CFG.debug:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    return model