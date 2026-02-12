import cv2
import numpy as np


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def reset_size_pred(masks, meta_normalization):
    """Resize predicted masks back to original image size.

    Args:
        masks: numpy array of shape (N, H, W) - predicted masks
        meta_normalization: list of metadata dictionaries for each image

    Returns:
        List of resized masks as numpy arrays
    """
    resized_masks = []
    if not isinstance(meta_normalization, list):
        meta_normalization = [meta_normalization]
    for i, meta in enumerate(meta_normalization):

        meta_resample = meta["resample"]
        orig_h, orig_w = meta_resample["original_size"]  # torch tensor
        orig_h = int(orig_h.item())
        orig_w = int(orig_w.item())
        if meta_resample["resize_mode"] == "pad_and_resize":
            pad_h, pad_w = meta_resample["pad"]
            resized_mask = masks[i][: orig_h + pad_h, : orig_w + pad_w]
        else:  # 'resize_only'
            resized_mask = cv2.resize(
                masks[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_CUBIC,
            )
        resized_masks.append(resized_mask)
    return resized_masks
