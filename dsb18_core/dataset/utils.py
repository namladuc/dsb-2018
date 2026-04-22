import cv2
import numpy as np
import torch


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
        meta_normalization: list of metadata dictionaries or a single dict (batched)

    Returns:
        List of resized masks as numpy arrays
    """
    resized_masks = []
    
    # Determine number of samples
    if isinstance(meta_normalization, dict):
        # Determine N from masks shape or a known tensor in meta
        N = masks.shape[0] if len(masks.shape) == 3 else 1
    elif isinstance(meta_normalization, list):
        N = len(meta_normalization)
    else:
        meta_normalization = [meta_normalization]
        N = 1

    for i in range(N):
        if isinstance(meta_normalization, dict):
            # Extract sample meta from batched dict
            meta_resample = meta_normalization["resample"]
            orig_size = meta_resample["original_size"]
            
            # If orig_size is [Tensor(B), Tensor(B)] (standard collate for [h, w])
            if isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
                orig_h, orig_w = orig_size[0][i], orig_size[1][i]
            elif torch.is_tensor(orig_size) and orig_size.dim() == 2:
                orig_h, orig_w = orig_size[i]
            else:
                # Fallback
                orig_h, orig_w = orig_size[i] if hasattr(orig_size, "__getitem__") else (orig_size, orig_size)
            
            resize_mode = meta_resample["resize_mode"]
            if isinstance(resize_mode, (list, tuple)):
                resize_mode = resize_mode[i]
            
            if "pad" in meta_resample:
                pad = meta_resample["pad"]
                if isinstance(pad, (list, tuple)) and len(pad) == 2:
                    pad_h, pad_w = pad[0][i], pad[1][i]
                elif torch.is_tensor(pad) and pad.dim() == 2:
                    pad_h, pad_w = pad[i]
                else:
                    pad_h, pad_w = pad[i] if hasattr(pad, "__getitem__") else (0, 0)
            else:
                pad_h, pad_w = 0, 0
        else:
            meta = meta_normalization[i]
            meta_resample = meta["resample"]
            orig_h, orig_w = meta_resample["original_size"]
            resize_mode = meta_resample["resize_mode"]
            pad_h, pad_w = meta_resample.get("pad", (0, 0))

        # Convert torch tensors to scalars if needed
        real_h = int(orig_h.item()) if hasattr(orig_h, "item") else int(orig_h)
        real_w = int(orig_w.item()) if hasattr(orig_w, "item") else int(orig_w)
        
        mask_to_resize = masks[i] if len(masks.shape) == 3 else masks

        if resize_mode == "pad_and_resize":
            real_pad_h = int(pad_h.item()) if hasattr(pad_h, "item") else int(pad_h)
            real_pad_w = int(pad_w.item()) if hasattr(pad_w, "item") else int(pad_w)
            resized_mask = mask_to_resize[: real_h + real_pad_h, : real_w + real_pad_w]
        else:  # 'resize_only'
            resized_mask = cv2.resize(
                mask_to_resize,
                (real_w, real_h),
                interpolation=cv2.INTER_CUBIC,
            )
        resized_masks.append(resized_mask)
    return resized_masks
