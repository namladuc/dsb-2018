import albumentations as A 
from skimage.util import random_noise 
import numpy as np
import torch
import cv2 

def get_aug_dict(CFG):
    augmentation_dict = {
        'baseline': {
            'train': A.Compose([
                # dont do nothing
            ], p=1),
            'valid': A.Compose([
                # dont do nothing
            ], p=1),
        },
    }
    return augmentation_dict[CFG.aug]
    
class DropChannelRandom(A.ImageOnlyTransform):     
    def __init__(self, always_apply=False, p=0.5, name="DropChannelRandom", **kwargs): 
        super().__init__(always_apply, p)      
                   
    def apply(self, img, **params): 
        num_channel = img.shape[2]
        channel_choice = num_channel // 2
        while channel_choice == (num_channel // 2):
            channel_choice = np.random.randint(low = 0, high=num_channel)
        img[:,:,channel_choice] = np.zeros((img.shape[:2]), dtype=np.float32)
        return img
   