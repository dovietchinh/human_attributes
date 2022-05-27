import os
from utils.dataset import LoadImagesAndLabels,preprocess
import yaml
import pandas as pd
import logging
import argparse
import torch
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

def view_augment(opt):

    df_train = []
    if isinstance(opt.train_csv,str):
        opt.train_csv = [opt.train_csv]
    for file in opt.train_csv:
        df_train.append(pd.read_csv(file))
    df_train = pd.concat(df_train,axis=0)
    # a = df_train
    # b = a[:28918]
    # c = a[43051:]
    # df_train = pd.concat([b,c],axis=0)
    # df_train = df_train[df_train.body_pose==0]
    df_train = df_train[df_train.quality==1]
    print(len(df_train))
    ds_train = LoadImagesAndLabels(df_train,
                                data_folder=opt.TRAIN_FOLDER,
                                img_size = opt.img_size,
                                padding = opt.padding,
                                classes = opt.classes,
                                format_index = opt.format_index,
                                preprocess=preprocess,
                                augment=True,
                                augment_params=opt.augment_params)

    trainLoader = torch.utils.data.DataLoader(ds_train,
                                             batch_size=16,#opt.batch_size,
                                            shuffle=False,)
               
    for imgs,labels,_ in trainLoader:
        labels = labels.permute(1,0)
        
        # imgs = imgs.view(16,3,224,224)

        stack = [[],[],[],[]]                                 
        imgs = (imgs*255).type(torch.uint8).detach().cpu().numpy() #batch,3,H,W
        for j,img in enumerate(imgs):
            img = np.transpose(img,[1,2,0])
            stack[j//4].append(img)
        for a in range(4):
            stack[a] = np.concatenate(stack[a],axis=1)    
        stack = np.concatenate(stack,axis=0)
        cv2.imshow('a',stack)
        k = cv2.waitKey(0)
        if k ==ord('q'):
            exit()



def parse_opt(know=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',help = 'weight path')
    parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/multi-task-classification/config/human_attribute_4/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/multi-task-classification/config/human_attribute_4/data_config.yaml')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30, help='patience epoch for EarlyStopping')
    parser.add_argument('--save_dir', type=str, default='', help='save training result')
    parser.add_argument('--task_weights', type=list, default=1, help='weighted for each task while computing loss')
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

if __name__ =='__main__': 
    opt = parse_opt(True)
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    for k,v in cfg.items():
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    assert isinstance(opt.classes,dict), "Invalid format of classes in data_config.yaml"
    # assert len(opt.task_weights) == len(opt.classes), "task weight should has the same length with classes"
    if opt.DEBUG:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)
    print(opt.padding)
    view_augment(opt)