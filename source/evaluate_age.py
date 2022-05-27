import torch
from models import MobileNetV2,resnet18,resnet34,resnet50,resnet101,resnet152,\
                    resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,\
                    wide_resnet101_2,shufflenet_v2_x0_5,shufflenet_v2_x1_0
from models import model_fn,model_urls
# from models.resnet_age import resnet50
import sklearn.metrics 
from tqdm import tqdm 
import argparse
import logging 
from utils.dataset_age import LoadImagesAndLabels, preprocess
from utils.torch_utils import select_device
import yaml
import pandas as pd
import os
import numpy as np

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
    df_val = df_val[df_val.age2!=-1]
    df_val = df_val.dropna()
    # df_val = df_val[df_val.body_pose!=2]
    # df_val = df_val[df_val.action==1]

    # import shutil
    # for i in tqdm(df_val.iloc,total=len(df_val)):
    #     age = i.age
    #     old_path = os.path.join(opt.TEST_FOLDER,i.path)
    #     new_path = os.path.join('aaaaaaaaa',str(age),os.path.basename(i.path))
    #     shutil.copy(old_path,new_path)
    # exit()

    # df_val = df_val[df_val.visible==0]
    # df_val = df.sample(frac=1).reset_index(drop_index=True)
    
    
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
    # model = resnet50(num_classes=75)
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
    device = torch.device(opt.device)
    model = model.to(device)
    model.eval()
    rank = torch.Tensor([i for i in range(75)]).to(device)
    y_true = [] 
    y_pred = [] 
    for _ in range(len(opt.classes)):
    #     y_true.append([])
        y_pred.append([])
    with torch.no_grad(): 
        for i,(imgs,labels,labels_2,path) in tqdm(enumerate(loader['val']),total=len(loader['val'])):
            imgs = imgs.to(device)
            preds = model.predict(imgs)
            # labels = labels.permute(1,0)
            labels = [label.to(device).cpu().numpy().ravel() for label in labels]
            # LOGGER.info(f'len_labels: {len(labels[0])}')
            # preds = [x.detach().cpu().numpy().argmax(dim=-1).ravel() for x in preds]
            # print(preds.shape)
            # print(preds[0].sum())
            # preds = preds* rank

            # print(preds.shape)
            # print(preds.argmax(axis=-1)[0])
            # exit()

            # preds = [x.detach().cpu().numpy().sum().ravel() for x in preds]
            preds = [x.detach().cpu().numpy().argmax(axis=-1).ravel() for x in preds]
            # print(labels.shape)
            # print(preds.shape)
            # print((labels))
            # print((preds))
            # exit()
            for j in range(len(opt.classes)):
                # print(labels[j].shape)
                # print(preds[j].shape)
                # exit()
                # y_true[j].append(labels[j])
                y_pred[j].append(preds[j])
            y_true.append(labels)
            # y_pred.append(preds)
    # y_true = [ np.concatenate(x, axis=0) for x in y_true ]
    y_pred = [ np.concatenate(x, axis=0) for x in y_pred ]
    y_true = np.concatenate(y_true,axis=0)
    # y_pred = np.concatenate(y_pred,axis=0)
    print(y_true.shape)
    print(y_pred[0].shape)
    mylabel = ['0-20','21-25','26-30','31-34','35','36-40','41-49','50+']
    for abc,egf in zip(y_true.reshape(-1),y_pred[0].reshape(-1)):
        egf = mylabel[egf]
        with open('abc.txt','a') as f:
            f.write(f"{abc:.02f} {egf}\n")

    np.save('y_true.npy',y_true)
    np.save('y_pred.npy',y_pred)
    result = np.abs(y_true-y_pred)/y_true
    result = result.mean()
    print(result)
    exit()

    # print(np.abs(y_true-y_pred))
    # exit()
    # LOGGER.debug(f"y_true[0]_len = {len(y_true[0])}")
    # LOGGER.debug(f"y_true[1]_len = {len(y_true[1])}")
    # LOGGER.debug(f"y_pred[0]_len = {len(y_pred[0])}")
    # LOGGER.debug(f"y_pred[1]_len = {len(y_true[1])}")
    for i,(k,v) in enumerate(opt.classes.items()):
        y_true_i = y_true[i]
        y_pred_i = y_pred[i]
        y_pred_i = y_pred_i[y_true_i!=-1]
        y_true_i = y_true_i[y_true_i!=-1]
        # print(f'-------------{k}-----------\n')
        # print(y_true_i.shape)
        # print(y_pred_i.shape)
        # print(np.unique(y_true_i))
        # print(np.unique(y_pred_i))
        
        
        fi = sklearn.metrics.classification_report(y_true_i,y_pred_i,digits=4,zero_division=1,target_names=v)
        with open(opt.logfile,'a') as f:
            f.write(f'-------------{k}-----------\n')
            f.write(fi+'\n')
            print(f'-------------{k}-----------\n')
            print(fi+'\n')


def parse_opt(know):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help="checkpoint path")
    parser.add_argument('--batch_size', type=int, default=3, help="batch size in evaluating")
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
        if k in ['device','batch_size']:
            continue
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    assert isinstance(opt.classes,dict), "Invalid format of classes in data_config.yaml"
    # assert len(opt.task_weights) == len(opt.classes), "task weight should has the same length with classes"
    # print(opt['batch_size'])
    evaluate(opt)

if __name__ =='__main__':
    main()



