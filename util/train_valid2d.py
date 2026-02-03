from Core import *
import wandb
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
from .loss import criterion, dice_coef, iou_coef, criterion_Tversky
def train_one_epoch2d(model, optimizer, scheduler, dataloader, device, CFG):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss   = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:        
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        optimizer.zero_grad()
        batch_size = images.size(0)
        
        y_pred = model(images)
        loss = 0
        if CFG.isDeeply:
            for index, predict_y in enumerate(y_pred): # deeply
                l = criterion(predict_y, masks, CFG)
                loss  += l / len(y_pred)
        else:
            loss = criterion(y_pred, masks, CFG)
            
        loss.backward()
        optimizer.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        
        if CFG.debug:
            break
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch2d(model, dataloader, device, optimizer, CFG):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss   = 0.0
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_preds = model(images)
        if (CFG.isDeeply):
            y_pred = y_preds[0]
        else:
            y_pred = y_preds
        loss    = criterion(y_pred, masks, CFG)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred, CFG).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred, CFG).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
        if CFG.debug:
            print()
            print("Input shape: ", images.shape)
            print("Output shape: ", y_pred.shape)
            break
        
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training2d(model, optimizer, scheduler, run, num_epochs, train_loader, valid_loader, CFG):  
    save_dir = os.getcwd() + "/runs"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sett_dir = save_dir + f"/{CFG.dataset}_{CFG.net_structure}_{CFG.backbone}_{CFG.model_name}_{CFG.aug}"
    if not os.path.exists(sett_dir):
        os.mkdir(sett_dir)
    exp_dir = sett_dir + f"/{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        
    if (CFG.using_wandb == 1):
        # To automatically log gradients
        wandb.watch(model, log_freq=100)
        
    if (CFG.resume_train):
        for count_e in range(CFG.epochs_res):
            scheduler.step()
    else:
        CFG.epochs_res = 1
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    if CFG.resume_train:
        best_dice = CFG.best_dice
        best_epoch = CFG.best_epoch
    history = defaultdict(list)
    
    for epoch in range(CFG.epochs_res, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch2d(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, CFG=CFG)
        
        val_loss, val_scores = valid_one_epoch2d(model, valid_loader, 
                                                 device=CFG.device, optimizer=optimizer, CFG=CFG)
        
        scheduler.step()
        val_dice, val_jaccard = val_scores
    
        if CFG.debug:
            break
        
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        if (CFG.using_wandb == 1):
            # Log the metrics
            wandb.log({"Train Loss": train_loss, 
                    "Valid Loss": val_loss,
                    "Valid Dice": val_dice,
                    "Valid Jaccard": val_jaccard,
                    "LR":scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice > best_dice:
            print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            
            if (CFG.using_wandb == 1):
                run.summary["Best Dice"]    = best_dice
                run.summary["Best Jaccard"] = best_jaccard
                run.summary["Best Epoch"]   = best_epoch
                
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"/best_epoch{CFG.model_name}-{CFG.fold_selected:02d}.bin"
            torch.save(model.state_dict(),exp_dir + PATH)
            # Save a model file from the current 
            if (CFG.using_wandb == 1):
                wandb.save(exp_dir + PATH, base_path = exp_dir)
                print(f"Model Saved")
        
        PATH = f"/last_epoch{CFG.model_name}-{CFG.fold_selected:02d}.bin"
        torch.save(model.state_dict(),exp_dir + PATH)
        if (CFG.using_wandb == 1):
            wandb.save(exp_dir + PATH, base_path = exp_dir)
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history