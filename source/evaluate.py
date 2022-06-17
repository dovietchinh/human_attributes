import torch
from models import MobileNetV2,resnet18,resnet34,resnet50,resnet101,resnet152,\
                    resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,\
                    wide_resnet101_2,shufflenet_v2_x0_5,shufflenet_v2_x1_0
from models import model_fn,model_urls
import sklearn.metrics 
from tqdm import tqdm 
import argparse
import logging 
from utils.dataset import LoadImagesAndLabels, preprocess
from utils.torch_utils import select_device
import yaml
import pandas as pd
import os
import numpy as np
import time 

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig()

def evaluate(opt):
    if not hasattr(opt, 'TEST_FOLDER'):
        opt.TEST_FOLDER = opt.TRAIN_FOLDER
    if isinstance(opt.test_csv,str):
        opt.test_csv = [opt.test_csv]
    df_val = []
    for df in opt.test_csv:
        df  = pd.read_csv(df)
        df_val.append(df)
    df_val = pd.concat(df_val, axis=0)
    
    if not os.path.isfile(opt.weights): 
        LOGGER.info(f"{opt.weights} is not a file")
        exit()
    checkpoint = torch.load(opt.weights)
    old_opt = checkpoint['meta_data']
    model_config = []
    for k,v in old_opt.classes .items():
        model_config.append(len(v))
    # init model
    model = model_fn[opt.model_name](model_config=model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    padding = getattr(checkpoint['meta_data'], 'padding')
    img_size = getattr(checkpoint['meta_data'], 'img_size')
    
    
    ds_val = LoadImagesAndLabels(df_val,
                                data_folder=opt.TEST_FOLDER,
                                img_size = img_size,
                                padding= padding,
                                classes = opt.classes,
                                format_index = opt.format_index,
                                preprocess=preprocess,
                                augment=False)
    loader_val = torch.utils.data.DataLoader(ds_val,
                                            batch_size=opt.batch_size,
                                            shuffle=True
                                            )
    loader = {'val':loader_val}
    device = select_device(opt.device, model_name=getattr(checkpoint['meta_data'],'model_name'))
    model = model.to(device)
    model.eval()
    criterior = torch.nn.L1Loss(reduction='mean')
    epoch_loss = 0.0 
    with torch.no_grad(): 
        for i,(imgs,labels,path) in tqdm(enumerate(loader['val']),total=len(loader['val'])):
            imgs = imgs.to(device)
            preds = model.predict(imgs)
            labels = labels.permute(1,0).to(device)
            loss = 0 
            for index in range(len(opt.classes)):
                loss += criterior(preds[index].view(-1),labels[index])
            
            #labels = [label.to(device).cpu().numpy().ravel() for label in labels]
            #preds = [x.detach().cpu().numpy().argmax(axis=-1).ravel() for x in preds]
            epoch_loss += loss * imgs.size(0)
    epoch_loss = epoch_loss / len(loader['val'].dataset)
    print(epoch_loss)    
    localtime = time.localtime()
    year = localtime.tm_year
    day = localtime.tm_mday
    month = localtime.tm_mon
    with open(os.path.join(opt.save_dir,'evaluate.txt'),'w') as f:
        f.write(f"{year}_{month}_{day} : {loss}\n")
        
def parse_opt(know):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help="checkpoint path")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size in evaluating")
    parser.add_argument('--device', type=str, default='', help="select gpu")
    parser.add_argument("--val_csv",type=str, default='',help='')
    parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/multi-task-classification/config/human_attribute_4/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/multi-task-classification/config/human_attribute_4/data_config.yaml')
    parser.add_argument("--logfile", type=str, default="log.evaluate.txt", help="log the evaluating result")
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

def main():
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
    evaluate(opt)

if __name__ =='__main__':
    main()



