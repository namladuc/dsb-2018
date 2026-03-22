import albumentations as A
from skimage.util import random_noise
import numpy as np
import torch
import cv2


def get_aug_dict(CFG):

    # - Clahe, Sharpen, Emboss
    # - Gaussian Noise
    # - Color to Gray
    # - Inverting - we should not have used it, some images were notpredicted correctly on stage2 because of this augmentation
    # - Remapping grayscale images to random color images
    # - Blur, Median Blur, Motion Blur
    # - contrast and brightness
    # - random scale, rotates and flips
    # - Heavy geometric transformations: Elastic Transform, PerspectiveTransform, Piecewise Affine transforms, pincushion distortion
    # - Random HSV
    # - Channel shuffle - I guess this one was very important due to thenature of the data
    # - Nucleus copying on images. That created a lot of overlapping nuclei.It seemed to help networks to learn better borders for overlappingnuclei.

    augmentation_dict = {
        "aug1": {
            "train": A.Compose(
                [
                    # Clahe, Sharpen, Emboss
                    A.OneOf(
                        [
                            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                            A.Sharpen(alpha=(0.2, 0.5)),
                            A.Emboss(alpha=(0.2, 0.5)),
                        ],
                        p=0.5,
                    ),
                    # Gaussian Noise
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(10.0, 50.0)),
                            A.ISONoise(color_shift=(0.01, 0.05)),
                        ],
                        p=0.5,
                    ),
                    # Blur, Median Blur, Motion Blur
                    A.OneOf(
                        [
                            A.Blur(blur_limit=7),
                            A.MedianBlur(blur_limit=7),
                            A.MotionBlur(blur_limit=7),
                        ],
                        p=0.5,
                    ),
                    # contrast and brightness
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    # random scale, rotates and flips
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=45,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Random HSV
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                    ),
                    A.ChannelShuffle(p=0.5),
                ],
                p=1,
            ),
            "valid": A.Compose(
                [
                    # dont do nothing
                ],
                p=1,
            ),
        },
        "baseline": {
            "train": A.Compose(
                [
                    # dont do nothing
                ],
                p=1,
            ),
            "valid": A.Compose(
                [
                    # dont do nothing
                ],
                p=1,
            ),
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
            channel_choice = np.random.randint(low=0, high=num_channel)
        img[:, :, channel_choice] = np.zeros((img.shape[:2]), dtype=np.float32)
        return img
