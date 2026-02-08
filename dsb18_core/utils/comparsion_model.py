from dsb18_core import *
import pandas as pd
import numpy as np # linear algebra
from tqdm import tqdm
tqdm.pandas()
import gc
import torch # PyTorch 
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.cuda import amp
import time
import copy
from collections import defaultdict
import os
from datetime import datetime
from .loss import criterion, dice_coef, iou_coef
from .metrics import hausdorff_slice_first as hausdorff
import timeit

def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    threshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ["Large Bowel", "Small Bowel", "Stomach"]) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).astype(np.float64)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
               truth: np.ndarray,
               threshold: float = 0.5,
               eps: float = 1e-9,
               classes: list = ["Large Bowel", "Small Bowel", "Stomach"]) -> np.ndarray:
    """
    Calculate Jaccard index for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).astype(np.float64)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return [true_positives, false_positives, true_negatives, false_negatives]

@torch.no_grad()
def valid_one_epoch2d(model, dataloaders, CFG, post_processing=False, thr_post=50):
    
    dataset_size     = 0
    running_loss     = 0.0
    epoch_loss       = 0.0
    
    total_time_infer = 0.0
    avg_time_infer   = 0.0
    count_infer      = 0.0
    
    total_case_infer = 0.0
    avg_case_infer   = 0.0
    count_infer_case = 0.0
    val_scores       = []
    list_case_name   = []
    precision_scores = []
    recall_scores    = []
    classes          = ["Large Bowel", "Small Bowel", "Stomach"]
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes  = {key: list() for key in classes}
    
    y_pred_thr_array = np.array([])
    y_true_thr_array = np.array([])
    
    pbar = tqdm(enumerate(dataloaders), total=len(dataloaders), desc='Valid ')
    for _, (dataloadelm, name_case) in pbar: 
        max_dist = np.sqrt(np.sum([x ** 2 for x in [CFG.img_size[0], CFG.img_size[1], len(dataloadelm.dataset)]]))       
        list_case_name.append(name_case)
        start_case = total_time_infer
        for _, (images, masks) in enumerate(dataloadelm):       
            
            images           = images.to(CFG.device, dtype=torch.float)
            masks            = masks.to(CFG.device, dtype=torch.float)
            batch_size       = images.size(0)
            
            with torch.no_grad():
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                y_preds        = model(images)
                ender.record()
                torch.cuda.synchronize()
                curr_time      = starter.elapsed_time(ender)/1000
                
            count_infer      += 1
            total_time_infer += curr_time
            avg_time_infer   = total_time_infer / count_infer
            
            if (CFG.isDeeply):
                y_pred = y_preds[0]
            else:
                y_pred = y_preds
            loss         = criterion(y_pred, masks, CFG)
            
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            epoch_loss   = running_loss / dataset_size
            
            y_pred       = nn.Sigmoid()(y_pred)
            
            if post_processing:
                for index_slice in range(y_pred.shape[0]):
                    if np.sum((y_pred[index_slice] > 0.5).to(torch.float32)) <= thr_post:
                        y_pred[index_slice] = torch.from_numpy(np.zeros(y_pred.shape[1:], dtype=np.float32))
                    
            val_dice     = dice_coef(masks, y_pred, CFG).cpu().detach().numpy()
            val_jaccard  = iou_coef(masks, y_pred, CFG).cpu().detach().numpy()
            val_hausdorff= hausdorff(masks, y_pred, max_dist)
            val_scores.append([val_dice, val_jaccard, val_hausdorff])
            
            # False positive
            y_pred_thr = (y_pred > 0.5).to(torch.float32)
            y_pred_thr = torch.sum(y_pred_thr, dim=(1,2,3)).cpu().detach().numpy() > 0
            y_true_thr = torch.sum(masks, dim=(1,2,3)).cpu().detach().numpy() > 0
            y_pred_thr_array = np.concatenate((y_pred_thr_array, y_pred_thr), axis=0)
            y_true_thr_array = np.concatenate((y_true_thr_array, y_true_thr), axis=0)
            
            # calc precision and recall for pixel segmentation
            y_pred_thr = (y_pred > 0.5).to(torch.float32)
            precision = torch.sum(y_pred_thr * masks) / (torch.sum(y_pred_thr) + 1e-6)
            recall    = torch.sum(y_pred_thr * masks) / (torch.sum(masks) + 1e-6)
            precision_scores.append(precision.cpu().detach().numpy())
            recall_scores.append(recall.cpu().detach().numpy())
            
            # per class
            dice_scores = dice_coef_metric_per_classes(
                y_pred.cpu().detach().numpy(), 
                masks.cpu().detach().numpy())
            
            iou_scores  = jaccard_coef_metric_per_classes(
                y_pred.cpu().detach().numpy(), 
                masks.cpu().detach().numpy())
            
            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])
            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])
                
        end_case = total_time_infer
        total_case_infer += end_case - start_case
        count_infer_case += 1
        avg_case_infer   = total_case_infer / count_infer_case
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(total_time=f'{total_time_infer:0.4f}',
                         time_per_case=f'{avg_case_infer:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    # CFS Matrix
    y_pred_thr_array = torch.tensor(y_pred_thr_array, dtype=torch.float32)
    y_true_thr_array = torch.tensor(y_true_thr_array, dtype=torch.float32)
    confusion_matrix = confusion(y_pred_thr_array, y_true_thr_array)    
    
    # Per class
    dice_df          = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns  = ['Large Bowel Dice', 'Small Bowel Dice', 'Stomach Dice']
    iou_df           = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns   = ['Large Bowel Jaccard', 'Small Bowel Jaccard', 'Stomach Jaccard']
    val_metics_df    = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df    = val_metics_df.loc[:, ['Large Bowel Dice', 'Large Bowel Jaccard', 
                                        'Small Bowel Dice', 'Small Bowel Jaccard', 
                                        'Stomach Dice', 'Stomach Jaccard']]
    val_mean_metric  = val_metics_df.mean()
    
    # Precision and Recall
    precision_scores = np.mean(precision_scores).astype(float)
    recall_scores    = np.mean(recall_scores).astype(float)
    
    return {
        "loss": epoch_loss,
        "dice_score": val_scores[0],
        "iou_score": val_scores[1],
        "hausdorff_score": val_scores[2],
        "total_time_infer": total_time_infer,
        "avg_time_infer": avg_time_infer,
        "total_case_infer": total_case_infer,
        "avg_case_infer": avg_case_infer,
        "confusion_matrix": confusion_matrix,
        "LB Dice": val_mean_metric['Large Bowel Dice'],
        "SB Dice": val_mean_metric['Small Bowel Dice'],
        "Stomach Dice": val_mean_metric['Stomach Dice'],
        "LB Jaccard": val_mean_metric['Large Bowel Jaccard'],
        "SB Jaccard": val_mean_metric['Small Bowel Jaccard'],
        "Stomach Jaccard": val_mean_metric['Stomach Jaccard'],
        "precision_pixel": precision_scores,
        "recall_pixel": recall_scores
    }
    
@torch.no_grad()
def valid_one_patient2d(model, dataloaders, CFG):     
    '''
        dataloader slice have mask only
    '''
    lb_metric = [0, 0, 0, 0] # 1
    sb_metric = [0, 0, 0, 0] # 2
    s_metric  = [0, 0, 0, 0] # 3
    no_metric = [0, 0, 0, 0] # 0
    for _, (images, masks) in enumerate(dataloaders):       
        images           = images.to(CFG.device, dtype=torch.float)
        masks            = masks.to(CFG.device, dtype=torch.float)
        batch_size       = images.size(0)
        
        with torch.no_grad():
            y_preds      = model(images)
        
        if (CFG.isDeeply):
            y_pred = y_preds[0]
        else:
            y_pred = y_preds
            
        y_pred       = nn.Sigmoid()(y_pred)
                
        y_pred_thr = (y_pred > 0.5).to(torch.float32)
        # calc confusion matrix for 3 class
        for i in range(batch_size):
            seg_pred  = y_pred[i].cpu().detach().numpy()
            seg_label = masks[i].cpu().detach().numpy()
            
            max_args = np.argmax(seg_pred, axis=0)
            max_v = np.max(seg_pred, axis=0)

            # Create a mask where max_v > 0.5
            threshold_mask = max_v > 0.5

            # Initialize an array with zeros
            new_seg_pred = np.zeros_like(seg_pred)

            # Set the positions of the max arguments to 1 where the threshold mask is true
            new_seg_pred[max_args, np.arange(320)[:, None], np.arange(384)] = threshold_mask

            # Copy the result back to seg_pred
            seg_pred[:, np.arange(320)[:, None], np.arange(384)] = new_seg_pred
            
            # 1 - large bowel
            lb_metric[0] += np.sum(seg_pred[0] * seg_label[0])
            lb_metric[1] += np.sum(seg_pred[0] * seg_label[1])
            lb_metric[2] += np.sum(seg_pred[0] * seg_label[2])
            lb_metric[3] += np.sum(np.where((seg_pred[0] == 1) & (seg_label[0] == 0) 
                                           & (seg_label[1] == 0) & (seg_label[2] == 0), 1, 0))
            
            # 2 - small bowel
            sb_metric[0] += np.sum(seg_pred[1] * seg_label[0])
            sb_metric[1] += np.sum(seg_pred[1] * seg_label[1])
            sb_metric[2] += np.sum(seg_pred[1] * seg_label[2])
            sb_metric[3] += np.sum(np.where((seg_pred[1] == 1) & (seg_label[0] == 0) 
                                           & (seg_label[1] == 0) & (seg_label[2] == 0), 1, 0))
            
            # 3 - stomach
            s_metric[0]  += np.sum(seg_pred[2] * seg_label[0])
            s_metric[1]  += np.sum(seg_pred[2] * seg_label[1])
            s_metric[2]  += np.sum(seg_pred[2] * seg_label[2])
            s_metric[3]  += np.sum(np.where((seg_pred[2] == 1) & (seg_label[0] == 0) 
                                           & (seg_label[1] == 0) & (seg_label[2] == 0), 1, 0))
            
            # 4 - none
            no_metric[0] += np.sum(np.where((seg_label[0] == 1) & (seg_pred[0] == 0) 
                                           & (seg_pred[1] == 0) & (seg_pred[2] == 0), 1, 0))
            no_metric[1] += np.sum(np.where((seg_label[1] == 1) & (seg_pred[0] == 0) 
                                           & (seg_pred[1] == 0) & (seg_pred[2] == 0), 1, 0))
            no_metric[2] += np.sum(np.where((seg_label[2] == 1) & (seg_pred[0] == 0) 
                                           & (seg_pred[1] == 0) & (seg_pred[2] == 0), 1, 0))
            no_metric[3] += np.sum(np.where((seg_label[0] == 0) & (seg_label[1] == 0) & 
                                            (seg_label[2] == 0) & (seg_pred[0] == 0) 
                                           & (seg_pred[1] == 0) & (seg_pred[2] == 0), 1, 0))
            
    return lb_metric, sb_metric, s_metric, no_metric
        

@torch.no_grad()
def valid_one_epoch3d(model, dataloaders, CFG): 
    model.eval()
    
    running_loss            = 0.0
    epoch_loss              = 0.0
    total_time_infer        = 0.0
    avg_time_infer          = 0.0
    total_case_infer        = 0.0
    avg_case_infer          = 0.0
    count_infer             = 0.0
    count_infer_case        = 0.0
    val_scores              = []
    gaussian_importance_map = _get_gaussian((CFG.num_slice, CFG.patch_size, CFG.patch_size), sigma_scale = 1. / 8)
    w_d, w_h, w_w           = (CFG.num_slice, CFG.patch_size, CFG.patch_size)
    count_elm               = 1
    classes                 = ["Large Bowel", "Small Bowel", "Stomach"]
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes  = {key: list() for key in classes}
    y_pred_thr_array        = np.array([])
    y_true_thr_array        = np.array([])
    
    pbar = tqdm(enumerate(dataloaders), total=len(dataloaders), desc='Valid ')
    for _, dataloadelm in pbar: 
        max_dist = np.sqrt(np.sum([x ** 2 for x in [CFG.img_size[0], CFG.img_size[1], len(dataloadelm.dataset)]])) 
        d, h, w, c = dataloadelm.dataset.getMetaPad()
        
        result = np.zeros((c, d, h, w), dtype='float32')
        overlap = np.zeros((c, d, h, w), dtype='float32')
        start_case = total_time_infer
        for _, (patchs, x, y, z) in enumerate(dataloadelm):
            patchs     = patchs.to(CFG.device, dtype=torch.float)
            x          = x.to('cpu', dtype=torch.float).detach().numpy()
            y          = y.to('cpu', dtype=torch.float).detach().numpy()
            z          = z.to('cpu', dtype=torch.float).detach().numpy()
            
            batch_size = patchs.size(0)
            
            with torch.no_grad():
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                y_preds        = model(patchs)
                ender.record()
                torch.cuda.synchronize()
                curr_time      = starter.elapsed_time(ender)/1000
                
            count_infer      += 1
            total_time_infer += curr_time
            avg_time_infer   = total_time_infer / count_infer
            
            if (CFG.isDeeply):
                y_pred = y_preds[0]
            else:
                y_pred = y_preds
            
            for i in range(batch_size):
                x_elm = int(x[i])
                y_elm = int(y[i])
                z_elm = int(z[i])
                if 'cuda' == CFG.device.type:
                    temp_arr = y_pred[i].cpu().detach().numpy()
                else:
                    temp_arr = y_pred[i].detach().numpy()
                result[:, x_elm:x_elm + w_d, y_elm:y_elm + w_h, z_elm:z_elm + w_w] += temp_arr * gaussian_importance_map
                overlap[:, x_elm:x_elm + w_d, y_elm:y_elm + w_h, z_elm:z_elm + w_w] += gaussian_importance_map
            
        end_case = total_time_infer
        total_case_infer += end_case - start_case
        count_infer_case += 1
        avg_case_infer   = total_case_infer / count_infer_case
        torch.cuda.empty_cache()
        gc.collect()
        
        masks = dataloadelm.dataset.getMaskAll()
        if not CFG.debug:
            assert np.sum(overlap == 0.) == 0, "Sliding window does not cover all volume"
        result = result / overlap
        
        # Batch, Channel, Dim, Height, Width
        masks         = masks[np.newaxis, :]
        result        = result[np.newaxis, :]
        
        result        = torch.tensor(result).to(CFG.device, dtype=torch.float)
        masks         = torch.tensor(masks).to(CFG.device, dtype=torch.float)
        
        loss          = criterion3D(result, masks, CFG)
        running_loss  += loss.item()
        
        y_pred        = nn.Sigmoid()(result)
        val_dice      = dice_coef(masks, y_pred, CFG, dim=(3, 4), mean_dim=(2, 1, 0)).cpu().detach().numpy()
        val_jaccard   = iou_coef(masks, y_pred, CFG, dim=(3, 4), mean_dim=(2, 1, 0)).cpu().detach().numpy()
        val_hausdorff = hausdorff(masks[0], y_pred[0], max_dist)
        val_scores.append([val_dice, val_jaccard, val_hausdorff])
        count_elm += 1
        
        # False positive
        y_pred_thr = (y_pred > 0.5).to(torch.float32)
        y_pred_thr = torch.sum(y_pred_thr[0], dim=(0,2,3)).cpu().detach().numpy() > 0
        y_true_thr = torch.sum(masks[0], dim=(0,2,3)).cpu().detach().numpy() > 0
        y_pred_thr_array = np.concatenate((y_pred_thr_array, y_pred_thr), axis=0)
        y_true_thr_array = np.concatenate((y_true_thr_array, y_true_thr), axis=0)
        
        # per class
        dice_scores = dice_coef_metric_per_classes(
            y_pred.cpu().detach().numpy(), 
            masks.cpu().detach().numpy())
        
        iou_scores  = jaccard_coef_metric_per_classes(
            y_pred.cpu().detach().numpy(), 
            masks.cpu().detach().numpy())
        for key in dice_scores.keys():
            dice_scores_per_classes[key].extend(dice_scores[key])
        for key in iou_scores.keys():
            iou_scores_per_classes[key].extend(iou_scores[key])
        
        dataloadelm.dataset.clearCache()
        torch.cuda.empty_cache()
        gc.collect()
        
        if CFG.debug:
            print("valid score: ", val_scores)
            print("loss: ", loss)
            print("Mask shape: ", masks.shape)
            break
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(total_time=f'{total_time_infer:0.4f}',
                         time_per_case=f'{avg_case_infer:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
    epoch_loss = running_loss / count_elm
    val_scores  = np.mean(val_scores, axis=0)
    
    # CFS Matrix
    y_pred_thr_array = torch.tensor(y_pred_thr_array, dtype=torch.float32)
    y_true_thr_array = torch.tensor(y_true_thr_array, dtype=torch.float32)
    confusion_matrix = confusion(y_pred_thr_array, y_true_thr_array)    
    
    # Per class
    dice_df         = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['Large Bowel Dice', 'Small Bowel Dice', 'Stomach Dice']
    iou_df          = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns  = ['Large Bowel Jaccard', 'Small Bowel Jaccard', 'Stomach Jaccard']
    val_metics_df   = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df   = val_metics_df.loc[:, ['Large Bowel Dice', 'Large Bowel Jaccard', 
                                        'Small Bowel Dice', 'Small Bowel Jaccard', 
                                        'Stomach Dice', 'Stomach Jaccard']]
    val_mean_metric = val_metics_df.mean()
    
    return {
        "loss": epoch_loss,
        "dice_score": val_scores[0],
        "iou_score": val_scores[1],
        "hausdorff_score": val_scores[2],
        "total_time_infer": total_time_infer,
        "avg_time_infer": avg_time_infer,
        "total_case_infer": total_case_infer,
        "avg_case_infer": avg_case_infer,
        "confusion_matrix": confusion_matrix,
        "LB Dice": val_mean_metric['Large Bowel Dice'],
        "SB Dice": val_mean_metric['Small Bowel Dice'],
        "Stomach Dice": val_mean_metric['Stomach Dice'],
        "LB Jaccard": val_mean_metric['Large Bowel Jaccard'],
        "SB Jaccard": val_mean_metric['Small Bowel Jaccard'],
        "Stomach Jaccard": val_mean_metric['Stomach Jaccard']
    }