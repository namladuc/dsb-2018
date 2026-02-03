import cv2
import numpy as np
import torch
from .util import rle_decode

def getDatasetValid2DInferCase(df, CFG, transforms):
    list_case_day    = df['case_day'].unique()
    list_ds = []
    for elm in list_case_day:
        subdf = df[df['case_day'] == elm].copy()
        subdf = subdf.reset_index(drop = True)
        list_ds.append([Dataset2D(subdf, CFG, "train", transforms), elm])
    return list_ds

class Dataset2D(torch.utils.data.Dataset):
    def __init__(self, df, CFG, subset="train", transforms=None):
        """[Dataset2D]
        Args:
            df (_type_): _DataFrame preprocessing for 25D meta data _
            CFG (_type_): _Config class_
            subset (str, optional): _'train' / 'test'_. Defaults to "train".
            transforms (_type_, optional): _Augmentation_. Defaults to None.
        """
        self.df = df
        self.subset = subset
        self.transforms = transforms
        
        self.width_norm = CFG.img_size[0]
        self.height_norm = CFG.img_size[1]
        self.lower_percentile = CFG.lower_percentile
        self.upper_percentile = CFG.upper_percentile
        self.num_slice = CFG.num_slice

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index): 
        img_paths=self.df['image_paths_25D'].iloc[index]
        w = self.df['width'].iloc[index]
        h = self.df['height'].iloc[index]
        
        # load image
        if self.num_slice <= 1:
            img_paths = self.df['path'].iloc[index]
            img = self.__load_an_img(img_paths)       
        else:
            img = self.__preprocess(img_paths, h, w)
        
        if len(img.shape) < 3:
            img = img[:,:,np.newaxis]
        
        # three label
        masks = np.zeros((h, w, 3), dtype=np.float32)
        if self.subset == 'train':
            for k,j in zip([0,1,2],["large_bowel","small_bowel","stomach"]):
                rles = self.df[j].iloc[index]
                mask = rle_decode(rles, shape=(h, w, 1))
                masks[:,:,k] = mask[:,:,0]
        
        if self.transforms:
            data = self.transforms(image=img, mask=masks)
            img  = data['image']
            masks  = data['mask']
            
        img = img.transpose(2, 0, 1)
        masks = masks.transpose(2, 0, 1)
        
        if self.subset == 'train': return torch.tensor(img), torch.tensor(masks)
        else: return torch.tensor(img), self.df['id'].iloc[index], h, w
    
    # image preprocessing step 1: read image and percentile clipping
    def __preprocess(self, img_paths, h, w):
        imgs = np.zeros((h, w, len(img_paths)), dtype=np.float32)
        for i in range(len(img_paths)):
            if img_paths[i] != "NAN":
                imgs[:,:,i] = self.__load_an_img(img_paths[i])                
            else:
                imgs[:,:,i] = self.__load_an_img(img_paths[len(img_paths) // 2])

        return imgs

    def __load_an_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Calculate intensity values corresponding to percentiles
        lower_value = np.percentile(img, self.lower_percentile)
        upper_value = np.percentile(img, self.upper_percentile)

        # Clip and normalize the pixel values
        clipped_image = np.clip(img, lower_value, upper_value)
        normalized_image = (clipped_image - lower_value) / (upper_value - lower_value)
        normalized_image = normalized_image.astype(np.float32)

        return normalized_image