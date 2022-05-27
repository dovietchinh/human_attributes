import torch
import os
import cv2
import yaml
import logging
from .augmentations import RandAugment
import numpy as np 
# from .imbalance_data_handle import balance_data
import pandas as pd
import random
LOGGER = logging.getLogger('__main__.'+__name__)

def preprocess(img,img_size,padding=True):
    if padding:
        height,width,_ = img.shape 
        delta = height - width 
        
        if delta > 0:
            img = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =0)
        else:
            img = np.pad(img,[[-delta//2,-delta//2],[0,0],[0,0]], mode='constant',constant_values =0)
    if isinstance(img_size,int):
        img_size = (img_size,img_size)
    try:    
        result = cv2.resize(img,img_size)
    except:
        result = img
        print(img.shape)
    return result

class LoadImagesAndLabels(torch.utils.data.Dataset):
    
    def __init__(self, csv, data_folder, img_size, padding, classes,format_index,preprocess=False, augment=False,augment_params=None):
        self.csv_origin = csv 
        self.data_folder = data_folder 
        self.augment = augment 
        self.preprocess = preprocess
        self.padding = padding
        self.img_size = img_size
        self.classes = classes
        if not format_index:
            self.maping_name = {}
            for k,v in classes.items():
                for index,classes_name in enumerate(v):
                    self.maping_name[classes_name] = index
        if augment:
            self.augmenter = RandAugment(augment_params=augment_params)
            # self.on_epoch_end(n=5000)     
            self.csv =self.csv_origin   
        else:
            self.csv =self.csv_origin
        # self.csv =self.csv_origin
        # print(self.maping_name)
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index,):
        item = self.csv.iloc[index]
        try:
            path = os.path.join(self.data_folder, item.path)
        except:
            print(item)
            exit()
        assert os.path.isfile(path),f'this image : {path} is corrupted'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            LOGGER.info(f' this image : {path} is corrupted')
        labels = []
        for label_name in self.classes:
            label = item[label_name]
            # label = self.maping_name[label]
            labels.append(label)

        if self.augment:
            img = self.augmenter(img)
        if self.preprocess:
            img = self.preprocess(img, img_size=self.img_size, padding=self.padding)
        img = np.transpose(img, [2,0,1])
        img = img.astype('float32')/255.
        return img,labels,path
            
    def on_epoch_end(self,n=500):
        csv = self.csv_origin
        labels = set(csv.age)
        dfs = []
        for label in labels:
            df = csv[csv.age==label].sample(n=n,replace=True)
            dfs.append(df)
        df = pd.concat(dfs,axis=0)
        df = df.sample(frac=1).reset_index(drop=True)
        self.csv =  df

