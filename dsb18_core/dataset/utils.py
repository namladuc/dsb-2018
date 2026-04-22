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
    return img.reshape(shape, order="F")  # Needed to align to RLE direction (column-major)


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def reset_size_pred(masks, meta_normalization):
    """Resize predicted masks back to original image size correctly.
    
    Handles:
    - Reversing center padding in 'pad_and_resize' mode
    - Reversing background cropping if 'crop_background' was used
    - Scaling back to original dimensions
    """
    import torch
    resized_masks = []
    
    # Normalize input to list of dicts for easier processing
    if isinstance(meta_normalization, dict):
        # Check if this is a batched dictionary (standard collate)
        # We assume it's batched if elements are tensors or lists of same length
        meta_list = []
        any_key = next(iter(meta_normalization))
        any_val = meta_normalization[any_key]
        
        # If any_val is a Tensor with batch dimension or a list, it's likely batched
        if torch.is_tensor(any_val) and any_val.dim() > 0:
            N = len(any_val)
            for i in range(N):
                sample_meta = {k: v[i] if torch.is_tensor(v) or isinstance(v, (list, tuple)) else v 
                             for k, v in meta_normalization.items()}
                meta_list.append(sample_meta)
        else:
            # Single sample dictionary
            meta_list = [meta_normalization]
    elif isinstance(meta_normalization, list):
        meta_list = meta_normalization
    else:
        meta_list = [meta_normalization]

    for i, meta in enumerate(meta_list):
        # Extract individual mask from batch
        if len(masks.shape) == 3:
            mask = masks[i]
        else:
            mask = masks

        # 1. Reverse Resampling/Padding
        if "resample" in meta:
            res = meta["resample"]
            # Helper to safely get int from possibly batched/tensor/list metadata
            def to_int(x):
                if hasattr(x, "item"): return int(x.item())
                if isinstance(x, (list, tuple, np.ndarray)): return int(x[0])
                return int(x)

            orig_h, orig_w = [to_int(x) for x in res["original_size"]]
            
            if res.get("resize_mode") == "pad_and_resize":
                # Get intermediate size and padding info
                new_h, new_w = [to_int(x) for x in res.get("new_size", (orig_h, orig_w))]
                pad_h_top, pad_w_left = [to_int(x) for x in res.get("pad", (0, 0))]
                
                # Crop center (reverse of center padding)
                mask = mask[pad_h_top : pad_h_top + new_h, pad_w_left : pad_w_left + new_w]
                # Resize back to original size (or size after crop if any)
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                # Direct resize
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Reverse Background Cropping
        if "crop" in meta:
            crop = meta["crop"]
            if crop.get("cropped", False):
                full_h, full_w = crop["original_shape"]
                full_h = int(full_h.item()) if hasattr(full_h, "item") else int(full_h)
                full_w = int(full_w.item()) if hasattr(full_w, "item") else int(full_w)
                
                bbox = crop["bbox"]
                y_min, y_max, x_min, x_max = [int(v.item()) if hasattr(v, "item") else int(v) for v in bbox]
                
                full_mask = np.zeros((full_h, full_w), dtype=mask.dtype)
                full_mask[y_min:y_max, x_min:x_max] = mask
                mask = full_mask
                
        resized_masks.append(mask)
        
    return resized_masks
