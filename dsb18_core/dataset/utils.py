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


def unbatch_meta(meta, batch_size=None):
    """Recursively unbatches metadata dictionaries from DataLoader.
    
    Returns a list of dictionaries (one per sample).
    """
    if batch_size is None:
        # Infer batch size from any tensor/list in the dict
        def get_bs(m):
            if torch.is_tensor(m): return len(m)
            if isinstance(m, (list, tuple)) and not isinstance(m[0], str): return len(m)
            if isinstance(m, dict):
                for v in m.values():
                    b = get_bs(v)
                    if b is not None: return b
            return None
        batch_size = get_bs(meta)
    
    if batch_size is None: return [meta]

    def get_idx(m, idx):
        if isinstance(m, dict):
            return {k: get_idx(v, idx) for k, v in m.items()}
        if torch.is_tensor(m):
            if m.dim() > 0 and len(m) == batch_size:
                return m[idx]
            return m
        if isinstance(m, (list, tuple)):
            # Special case: if this is a list/tuple of length batch_size,
            # and it contains simple types (not dicts/lists), it's likely the batch dimension.
            if len(m) == batch_size and not any(isinstance(v, (dict, list, tuple, torch.Tensor)) for v in m):
                return m[idx]
            # Otherwise, it might be a property (like size) containing batched tensors, so recurse.
            return [get_idx(v, idx) for v in m]
        return m
        
    return [get_idx(meta, i) for i in range(batch_size)]

def to_int(val):
    if hasattr(val, "item"): return int(val.item())
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0: return int(val[0])
    return int(val)

def reset_size_pred(masks, meta_normalization):
    """Resize predicted masks back to original image size correctly."""
    import torch
    
    if isinstance(meta_normalization, dict):
        meta_list = unbatch_meta(meta_normalization, len(masks))
    elif isinstance(meta_normalization, list):
        meta_list = meta_normalization
    else:
        meta_list = [meta_normalization] * len(masks)

    resized_masks = []
    for i in range(len(masks)):
        mask = masks[i]
        meta = meta_list[i]

        # 1. Reverse Resampling/Padding
        if "resample" in meta:
            res = meta["resample"]
            orig_h, orig_w = [to_int(x) for x in res["original_size"]]
            
            mode = res.get("resize_mode")
            if mode == "pad_and_resize":
                # Get intermediate size and padding info
                new_h, new_w = [to_int(x) for x in res.get("new_size", (orig_h, orig_w))]
                pad_h_top, pad_w_left = [to_int(x) for x in res.get("pad", (0, 0))]
                
                # Crop center (reverse of center padding)
                mask = mask[pad_h_top : pad_h_top + new_h, pad_w_left : pad_w_left + new_w]
                # Resize back to original size
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                # Direct resize or fallback
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Reverse Background Cropping
        if "crop" in meta:
            crop = meta["crop"]
            if crop.get("cropped", False):
                full_h, full_w = [to_int(x) for x in crop["original_shape"]]
                bbox = crop["bbox"]
                y_min, y_max, x_min, x_max = [to_int(v) for v in bbox]
                
                full_mask = np.zeros((full_h, full_w), dtype=mask.dtype)
                full_mask[y_min:y_max, x_min:x_max] = mask
                mask = full_mask
                
        resized_masks.append(mask)
        
    return resized_masks
