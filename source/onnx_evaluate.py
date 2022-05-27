import onnx
import numpy as np
import cv2
import onnxruntime as ort
import pandas as pd
import os
from tqdm import tqdm
import sklearn.metrics
from datetime import date
import shutil
BIN_FOLDER = 'images_bin'
RESULT_FOLDER = 'output_bin'
ATTRIBUTES = {
    # 'age'         : ['0-18', '18-55', '55+'], 
    'gender'      : ['female', 'male'],
    # 'shape'       : ['thin','fat','kid'],
    'hair'        : ['short_hair', 'long_hair'],
    'hat'         : ["no", "yes"],
    'glasses'     : ["no", "yes"],
    'face_mask'   : ["no", "yes"],
    'body_pose'   : ['front_pose', 'side_pose', 'turn_back_pose'],
    'visible'     : ['fullbody', 'upperbody'],
    'backpack'    : ["no", "yes"],
    'handbag'     : ["no", "yes"],
    'ub_length'   : ['short', 'long'],
    'lb_length'   : ['short', 'long'],
    # 'action'      : ['sitting','standing']
    # 'holding_object': ["no", "yes"]
    # 'crowd'       : ["yes","no"]

}
output_names = ['486','487','488','489','490','491','492','493','494','495','496']
  
LOGFILE = 'onnx_report.txt'
  

def sofmax(x):
    return np.exp(x)/sum(np.exp(x))

def save_bin():
    df = pd.read_csv('/u01/Intern/chinhdv/code/multi-task-classification/human_attribute_project/VCR_HMAT_test.csv')
    root = '/u01/Intern/chinhdv/github/auto_upload_cvat/hmat'
    for i in tqdm(df.iloc,total=len(df)):
        path = os.path.join(root,i.path)
        base_name = os.path.basename(i.path)
        shutil.copy(path,'images/'+base_name)

        
        

def test():
    df = pd.read_csv('/u01/Intern/chinhdv/code/multi-task-classification/human_attribute_project/VCR_HMAT_test.csv')
    y_true = {}
    y_pred = {}

    for k in ATTRIBUTES:
        y_true[k] = []
        y_pred[k] = []

    for i in tqdm(df.iloc,total=len(df)):
        
        base_name = os.path.basename(i.path)
        bin_path = '.'.join(base_name.split('.')[:-1]) 
        bin_path = os.path.join(RESULT_FOLDER,bin_path)
        # result = np.fromfile(bin_path,np.float32)
        if not os.path.isfile(bin_path+'_486.bin'):
                continue
        for index,(output_name,(name_attrbites,labels)) in enumerate(zip(output_names,ATTRIBUTES.items())):
            
            y_true[name_attrbites].append(i[name_attrbites])
            output = np.fromfile(bin_path+f'_{output_name}.bin',np.float32)
            # output = result[index]
            len_ = len(labels)
            output = sofmax(output[:len_])
            index_predicted = np.argmax(output) 
            # score = np.max(output) 
            y_pred[name_attrbites].append(index_predicted)
    
    with open(LOGFILE,'a') as f:
        today = str(date.today())
        f.write(f'.........{today}........\n')

    for name_attrbites,labels in ATTRIBUTES.items():
        y_pred[name_attrbites] = np.array(y_pred[name_attrbites])     
        y_true[name_attrbites] = np.array(y_true[name_attrbites])
        y_pred[name_attrbites] = y_pred[name_attrbites][y_true[name_attrbites] != -1]
        y_true[name_attrbites] = y_true[name_attrbites][y_true[name_attrbites] != -1]
        fi = sklearn.metrics.classification_report(y_true[name_attrbites],y_pred[name_attrbites],digits=4,zero_division=1,target_names=labels)
        with open(LOGFILE,'a') as f:
            f.write(f'-------------{name_attrbites}-----------\n')
            f.write(fi+'\n')
            print(f'-------------{name_attrbites}-----------\n')
            print(fi+'\n')

def abc():
    pass
    

if __name__ =='__main__':
    test()


# ort_sess = ort.InferenceSession('/u01/Intern/chinhdv/code/multi-task-classification/mobilenet_v2.onnx')
# outputs = ort_sess.run(None, {'input.1': x})
# print(outputs)
