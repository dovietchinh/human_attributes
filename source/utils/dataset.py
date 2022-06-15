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
        path = os.path.join(self.data_folder, item.path)
        assert os.path.isfile(path),f'this image : {path} is corrupted'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            LOGGER.info(f' this image : {path} is corrupted')
        labels = []
        # lb = 0
        # if random.random() > 0.5 and item.lb_length!= -1:
        #     lb = -1
        #     height = img.shape[0]
        #     img = img[:height//2,:,:]

        for label_name in self.classes:
            

            label = item[label_name]
            if label_name == 'age2':
                if label < 0:
                    label = -1
                if label >75:
                    label =75
            # label = self.maping_name[label]
            # if label_name == 'lb_length' and lb == -1:
            #     label = -1
            labels.append(label)

        # if random.random() > 0.5 and item.visible==0 and self.augment: #full_body
            # height = img.shape[0]
            # img = img[:height//2,:,:]

#        if self.augment:
#            img = self.augmenter(img)
        if random.random() > 0.5:
            img = np.fliplr(img)
        if self.preprocess:
            img = self.preprocess(img, img_size=self.img_size, padding=self.padding)
        img = np.transpose(img, [2,0,1])
        img = img.astype('float32')/255.
        # img = np.stack([img,img],axis=0)
        # print(len(labels))
        labels = torch.Tensor(labels).type(torch.float)
        labels = labels/255.
        
        # labels = [labels,labels]

        return img,labels,path

