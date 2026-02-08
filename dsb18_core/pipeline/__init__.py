from .runner_2d import run_training2d

def get_train_valid(CFG):
    if 'Unet2D' in CFG.net_structure:
        return run_training2d
    raise ValueError(f"Network structure '{CFG.net_structure}' not supported. "
                    f"This codebase supports 2D models only (DSB-2018 dataset).")