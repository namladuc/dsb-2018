import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
import platform

# Sklearn
from sklearn.model_selection import StratifiedGroupKFold

# PyTorch 
from torch.utils.data import DataLoader

# Albumentations for augmentations
import albumentations as A

# import dataset
from .Dataset.dataset2d import Dataset2D
from .Dataset.aug import getAugDict

def getDatasetMapping(CFG):
    if 'Unet25D' in  CFG.net_structure:
        return getTrainValidDatasetUWMadison25D(CFG, CFG.path_data)
    return None

def getTrainValidDatasetUWMadison25D(CFG, path_data):
    """
    CFG: Config class to config the dataset metadata
    path_data: Path to train dataset folder
    """
    
    # read csv
    df = pd.read_csv(path_data +  '/train.csv')
    df.rename(columns = {'class':'class_name'}, inplace = True)
    
    # split id column
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    
    # get image path
    TRAIN_DIR= path_data + "/train"
    all_train_images = glob(TRAIN_DIR + "/**/*.png", recursive=True)
    x = ""
    if platform.system() == "Windows":
        x = all_train_images[0].rsplit("\\", 4)[0] 
    else:
        x = all_train_images[0].rsplit("/", 4)[0]
        
    # path matching
    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(os.path.join(x,
                            "case"+str(df["case"].values[i]),
                            "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                            "scans",
                            "slice_"+str(df["slice"].values[i])))
    df["path_partial"] = path_partial_list
    
    path_partial_list = []
    for i in range(0, len(all_train_images)):
        path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))
    tmp_df = pd.DataFrame()
    tmp_df['path_partial'] = path_partial_list
    tmp_df['path'] = all_train_images
    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

    del x,path_partial_list,tmp_df
    
    # RESTRUCTURE  DATAFRAME
    df_train = pd.DataFrame({'id':df['id'][::3]})
    df_train['large_bowel'] = df['segmentation'][::3].values
    df_train['small_bowel'] = df['segmentation'][1::3].values
    df_train['stomach'] = df['segmentation'][2::3].values

    df_train['path'] = df['path'][::3].values
    df_train['case'] = df['case'][::3].values
    df_train['day'] = df['day'][::3].values
    df_train['slice'] = df['slice'][::3].values
    df_train['width'] = df['width'][::3].values
    df_train['height'] = df['height'][::3].values

    df_train.reset_index(inplace=True,drop=True)
    df_train.fillna('',inplace=True); 
    df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values
    
    # drop error case Only 7-day0 and 81-day30
    # https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/321979
    drop_index = df_train[
        ((df_train['case'] == 7) & (df_train['day'] == 0))  
        | ((df_train['case'] == 81) & (df_train['day'] == 30))
        | ((df_train['case'] == 43) & (df_train['day'] == 18))
        | ((df_train['case'] == 43) & (df_train['day'] == 26))
        | ((df_train['case'] == 138) & (df_train['day'] == 0))
        | ((df_train['case'] == 85) & (df_train['day'] == 23))
        | ((df_train['case'] == 90) & (df_train['day'] == 29))
        ].index
    df_train = df_train.drop(drop_index).reset_index(drop = True)
    
    if CFG.debug:
        df_train = df_train[(df_train['case'] == 2) | (df_train['case'] == 6)].reset_index(drop = True)
        
    assert(df_train.shape[0] != 0), "Error the training image data frame is empty!"
    
    channel_count = CFG.num_slice
    stride = CFG.stride
    count_index = 0
    for channel in range(- (channel_count // 2), (channel_count // 2) + 1, 1):
        df_train[f'image_path_{count_index:02}'] = df_train.groupby(
            ['case', 'day']
            )['path'].shift(-channel * stride).fillna(value="NAN")
        count_index += 1
    df_train['image_paths_25D'] = df_train[[f'image_path_{i:02d}' for i in range(channel_count)]].values.tolist()

    # count empty - create fold by case
    df_train['empty'] = (df_train['large_bowel']=='') & (df_train['small_bowel']=='') & (df_train['stomach']=='')
    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['empty'], groups = df_train["case"])):
        df_train.loc[val_idx, 'fold'] = fold
        
    data_transforms = getAugDict(CFG)
    train_df = df_train[df_train['fold'] != (CFG.fold_selected - 1)].reset_index(drop=True)
    valid_df = df_train[df_train['fold'] == (CFG.fold_selected - 1)].reset_index(drop=True)
    if (CFG.debug):
        print("Train DataFrame Shape: ", train_df.shape)
        print("Valid DataFrame Shape: ", valid_df.shape)
        
    train_dataset = Dataset2D(
        # df_train, # train all setting
                                train_df,
                               CFG,
                               transforms=data_transforms['train'])
    valid_dataset = Dataset2D(valid_df,
                               CFG,
                               transforms=data_transforms['valid'])
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, 
                              num_workers=CFG.numWorker, shuffle=True,
                              pin_memory=CFG.isPinMemory, drop_last=False)
    
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, 
                              num_workers=CFG.numWorker, shuffle=False,
                              pin_memory=CFG.isPinMemory)
    
    return train_loader, valid_loader

def getTrainValidTestDatasetUWMadison25D(CFG, path_data):
    """
    CFG: Config class to config the dataset metadata
    path_data: Path to train dataset folder
    """
    
    # read csv
    df = pd.read_csv(path_data +  '/train.csv')
    df.rename(columns = {'class':'class_name'}, inplace = True)
    
    # split id column
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    
    # get image path
    TRAIN_DIR= path_data + "/train"
    all_train_images = glob(TRAIN_DIR + "/**/*.png", recursive=True)
    x = ""
    if platform.system() == "Windows":
        x = all_train_images[0].rsplit("\\", 4)[0] 
    else:
        x = all_train_images[0].rsplit("/", 4)[0]
        
    # path matching
    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(os.path.join(x,
                            "case"+str(df["case"].values[i]),
                            "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                            "scans",
                            "slice_"+str(df["slice"].values[i])))
    df["path_partial"] = path_partial_list
    
    path_partial_list = []
    for i in range(0, len(all_train_images)):
        path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))
    tmp_df = pd.DataFrame()
    tmp_df['path_partial'] = path_partial_list
    tmp_df['path'] = all_train_images
    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

    del x,path_partial_list,tmp_df
    
    # RESTRUCTURE  DATAFRAME
    df_train = pd.DataFrame({'id':df['id'][::3]})
    df_train['large_bowel'] = df['segmentation'][::3].values
    df_train['small_bowel'] = df['segmentation'][1::3].values
    df_train['stomach'] = df['segmentation'][2::3].values

    df_train['path'] = df['path'][::3].values
    df_train['case'] = df['case'][::3].values
    df_train['day'] = df['day'][::3].values
    df_train['slice'] = df['slice'][::3].values
    df_train['width'] = df['width'][::3].values
    df_train['height'] = df['height'][::3].values

    df_train.reset_index(inplace=True,drop=True)
    df_train.fillna('',inplace=True); 
    df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values
    
    # drop error case Only 7-day0 and 81-day30
    # https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/321979
    drop_index = df_train[
        ((df_train['case'] == 7) & (df_train['day'] == 0))  
        | ((df_train['case'] == 81) & (df_train['day'] == 30))
        | ((df_train['case'] == 43) & (df_train['day'] == 18))
        | ((df_train['case'] == 43) & (df_train['day'] == 26))
        | ((df_train['case'] == 138) & (df_train['day'] == 0))
        | ((df_train['case'] == 85) & (df_train['day'] == 23))
        | ((df_train['case'] == 90) & (df_train['day'] == 29))
        ].index
    df_train = df_train.drop(drop_index).reset_index(drop = True)
    
    if CFG.debug:
        df_train = df_train[(df_train['case'] == 2) | (df_train['case'] == 6)].reset_index(drop = True)
        
    assert(df_train.shape[0] != 0), "Error the training image data frame is empty!"
    
    channel_count = CFG.num_slice
    stride = CFG.stride
    count_index = 0
    for channel in range(- (channel_count // 2), (channel_count // 2) + 1, 1):
        df_train[f'image_path_{count_index:02}'] = df_train.groupby(
            ['case', 'day']
            )['path'].shift(-channel * stride).fillna(value="NAN")
        count_index += 1
    df_train['image_paths_25D'] = df_train[[f'image_path_{i:02d}' for i in range(channel_count)]].values.tolist()

    # count empty - create fold by case
    df_train['empty'] = (df_train['large_bowel']=='') & (df_train['small_bowel']=='') & (df_train['stomach']=='')
    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['empty'], groups = df_train["case"])):
        df_train.loc[val_idx, 'fold'] = fold
        
    data_transforms = getAugDict(CFG)
    train_df = df_train[(df_train['fold'] != (CFG.fold_selected - 1)) & (df_train['fold'] != (CFG.fold_test - 1))].reset_index(drop=True)
    valid_df = df_train[df_train['fold'] == (CFG.fold_selected - 1)].reset_index(drop=True)
    test_df  = df_train[df_train['fold'] == (CFG.fold_test - 1)].reset_index(drop=True)
    
    if (CFG.debug):
        print("Train DataFrame Shape: ", train_df.shape)
        print("Valid DataFrame Shape: ", valid_df.shape)
        
    train_dataset = Dataset2D(
        # df_train, # train all setting
                                train_df,
                               CFG,
                               transforms=data_transforms['train'])
    valid_dataset = Dataset2D(valid_df,
                               CFG,
                               transforms=data_transforms['valid'])
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, 
                              num_workers=CFG.numWorker, shuffle=True,
                              pin_memory=CFG.isPinMemory, drop_last=False)
    
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, 
                              num_workers=CFG.numWorker, shuffle=False,
                              pin_memory=CFG.isPinMemory)
    
    test_loader = DataLoader(
        Dataset2D(test_df, CFG, transforms=data_transforms['valid']),
        batch_size=CFG.valid_bs, num_workers=CFG.numWorker, shuffle=False,
                              pin_memory=CFG.isPinMemory)
    
    return train_loader, valid_loader, test_loader
    