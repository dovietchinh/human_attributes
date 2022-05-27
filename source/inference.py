from mobilenetv2 import MobileNetV2
# from utils.dataset_elevator import LoadImagesAndLabels, preprocess
import torch
import cv2
import numpy as np
import os


classes = {
  # age         : [0-18, 18-55, 55+] 
  'gender'      : ['female', 'male'],
#   shape       : [thin,fat,kid]
  # hair        : [short_hair, long_hair]
  # hat         : ["no", "yes"]
  # glasses     : ["no", "yes"]
  # face_mask   : ["no", "yes"]
  # body_pose   : [front_pose, side_pose, turn_back_pose]
  # visible     : [fullbody, upperbody]
  # backpack    : ["no", "yes"]
  # handbag     : ["no", "yes"]
  'ub_length'   : ['short', 'long'],
  'lb_length'   : ['short', 'long'],
  'staff'       : ["no", 'SalesAssistant', 'TechnicalStaff'],
  # action      : ['sitting','standing']
  # holding_object: ["no", "yes"]
#   crowd       : ["yes","no"]
}


## see data_config.yaml

if __name__ =='__main__':
    model_config = []
    for k,v in classes.items():
        model_config.append(len(v))

    model = MobileNetV2(model_config =model_config)
    ckp = torch.load('/u01/Intern/chinhdv/code/multi-task-classification/result/runs_human_attributes_12/best.pt')
    model.load_state_dict(ckp['state_dict'])
    device = 'cuda:0'
    model = model.to(device)
    model.eval()
    path = '/u01/Intern/chinhdv/code/multi-task-classification/1_test7.jpg'
    img_raw = cv2.imread(path)
    
    img = cv2.resize(img_raw,(224,224))
    img = np.transpose(img,(2,0,1))
    img = img[None]
    img = img.astype('float')/255.
    img = torch.Tensor(img).to(device)
    outputs = model.predict(img)
    for index,(k,v) in enumerate(classes.items()):
        class_index = outputs[index].argmax(dim=-1)
        result = classes[k][class_index]
        print(f'{k} : {result}')
    # score = torch.max(output).numpy()
    # class_index = torch.argmax(output).numpy()
    
    # print(score,class_index)
    # cv2.imshow('a',img_raw)
    # k = cv2.waitKey(0)
    # if k ==ord('q'):
    #     exit()