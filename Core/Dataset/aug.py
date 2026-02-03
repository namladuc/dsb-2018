import albumentations as A 
from skimage.util import random_noise 
import numpy as np
import torch
import cv2 

def getAugDict(CFG):
    augmentation_dict = {
        'baseline': {
            'train': A.Compose([
                A.PadIfNeeded(min_height=CFG.img_size[1], min_width=CFG.img_size[0],
                              border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.,
                              position='top_left', p=1.0),
            ], p=1),
            'valid': A.Compose([
                A.PadIfNeeded(min_height=CFG.img_size[1], min_width=CFG.img_size[0],
                              border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.,
                              position='top_left', p=1.0),
            ], p=1),
        },
        'kit1': {
            "train":
                A.Compose([
                    A.PadIfNeeded(min_height=CFG.img_size[1], min_width=CFG.img_size[0],
                              border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.,
                              position='top_left', p=1.0),             
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),                            
                    A.GridDistortion(p=0.3),
                    A.RandomGamma(p=0.3),
                ], p=1.0),
            "valid": A.Compose([
                    A.PadIfNeeded(min_height=CFG.img_size[1], min_width=CFG.img_size[0],
                                border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.,
                                position='top_left', p=1.0),
                ], p=1.0)
        },
        'baseline_3d': {
            "train":
                A.Compose([
                   A.RandomCrop(height=CFG.patch_size, width=CFG.patch_size, p=1.0)
                ]),
            "valid":
                A.Compose([
                    
                ]),
        },
        'kit1_3d': {
            "train": Augment3D(CFG.num_slice, CFG.patch_size, CFG.patch_size, CFG.num_classes),
            "valid":
                A.Compose([
                    
                ]),
        }
    }
    return augmentation_dict[CFG.aug]
 
class PoissonNoiseSolution1(A.ImageOnlyTransform): 
    def __init__(self, always_apply=False, p=0.5, lambd=1.0, name="PoissonNoise", **kwargs): 
        super().__init__(always_apply, p) 
        self.lambd = lambd 
 
    def apply(self, img, **params): 
        return random_noise(img, mode="poisson")
    
class PoissonNoiseSolution2(A.ImageOnlyTransform):     
    def __init__(self, always_apply=False, p=0.5, lambd=1.0, name="PoissonNoise", **kwargs): 
        super().__init__(always_apply, p)         
        self.lambd = lambd 
        
    def apply(self, img, **params): 
        noise_mask = np.random.poisson(lam=self.lambd, size=img.shape)
        noisy_img = (img + noise_mask).astype(np.uint8)        
        return noisy_img
    
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
    
class Augment3D:
    '''
        Performs data augmentation on 3D data
    '''
    def __init__(self, image_depth, image_width, image_height, num_class):
        keys = {f'image{i}' : 'image' for i in range(1, image_depth)}
        keys.update({f'mask{i}' : 'mask' for i in range(1, image_depth)})
        self.transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(p=0.5),
                            A.GridDistortion(p=0.3),
                            A.RandomGamma(p=0.3)
                        ], additional_targets=keys)
                        
        self.d = image_depth
        self.w = image_width
        self.h = image_height
        self.num_class = num_class
    
    def __call__(self, image, mask):
        # Generate random patches 
        h, w, d  = image.shape[:3]
        start_d = start_h = start_w = 0
        if d > self.d:
            start_d = np.random.randint(0, d - self.d)
        if h > self.h:
            start_h = np.random.randint(0, h - self.h)
        if w > self.w:
            start_w = np.random.randint(0, w - self.w)

        image = image[start_h:start_h + self.h, start_w:start_w + self.w, start_d:start_d + self.d]
        mask = mask[start_h:start_h + self.h, start_w:start_w + self.w, start_d:start_d + self.d]

        # Add channel axis if necessary
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=-1)

        data = {f'image{i}' : image[:,:,i,:] for i in range(1, image.shape[2])}
        data.update({f'mask{i}' : mask[:,:,i,:] for i in range(1, image.shape[2])})
        data["image"] = image[:,:,0,:]
        data["mask"] = mask[:,:,0,:]

        aug_data = self.transform(**data)

        aug_image = np.empty(image.shape, dtype='float32')
        aug_image[:,:,0,:] = aug_data["image"]
        aug_mask = np.empty(mask.shape, dtype='float32')
        aug_mask[:,:,0,:] = aug_data["mask"]
        for i in range(1, image.shape[2]):
            aug_image[:,:,i,:] = aug_data[f"image{i}"]
            aug_mask[:,:,i,:] = aug_data[f"mask{i}"]
        return {"image": aug_image,"mask": aug_mask}