import torch
import cv2
import numpy as np
import os
import shutil
import argparse
import yaml
from tqdm import tqdm
from models import model_fn,model_urls

def get_parser():
    arg = argparse.ArgumentParser()
    arg.add_argument('--video',type=str,help='video path')
    arg.add_argument('--model_path',type=str,help='')
    arg.add_argument('--data',type=str,help='path to data config')
    arg.add_argument('--cfg',type=str,help='path to train config')
    arg.add_argument('--save_video',type=str,help='path_to_save_video')
    opt = arg.parse_args()
    return opt

class VideoReader():

    def __init__(self,file):
        self.cap = cv2.VideoCapture(file)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_index = 0
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self,):
        self.frame_index = 0
        return self

    def __len__(self,):
        return self.total_frame

    def __next__(self,):
        ret,frame = self.cap.read()
        self.frame_index +=1
        if not ret:
            raise StopIteration
        return ret,frame,self.frame_index

def box_label(img, box,line_width=None, label='', color=(0, 0, 255), txt_color=(0, 255, 0)):
    # Add one xyxy box to image with label
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    img = cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        outside = False
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # img = cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled

        # img = cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
        #             thickness=tf, lineType=cv2.LINE_AA)
        text = label

        text_size = w,h
        line_height = h + 5
        x,y0 = (p1[0], p1[1] - 2 if outside else p1[1] + h + 2)
        for i, line in enumerate(text.split("\n")):
            y = y0 + i * line_height
            cv2.putText(img,
                        line,
                        (x, y),
                        0,
                        # lw/6,
                        1,
                        color=txt_color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
    return img
def demo(opt):
    # os.system('cd /u01/Intern/chinhdv/code/Yolov5_DeepSort_Pytorch')
    os.system('rm -rf /u01/Intern/chinhdv/code/multi-task-classification/temp')
    os.system('cd /u01/Intern/chinhdv/code/Yolov5_DeepSort_Pytorch;python3 track.py --source {} --deep_sort_model shufflenet --yolo_model weights/crowdhuman_yolov5m.pt --classes 0 --device 0 --save-vid --save-txt --project /u01/Intern/chinhdv/code/multi-task-classification --name temp'.format(opt.video))
    # exit()

    model_config = []
    for k,v in opt.classes.items():
        model_config.append(len(v))

    # init model
    model = model_fn[opt.model_name](model_config=model_config)
    model.load_state_dict(torch.load(opt.model_path)['state_dict'])
    model = model.to('cuda:0')

    txt_name = os.path.basename(opt.video)
    txt_name = '.'.join(txt_name.split('.')[:-1])+'.txt'
    file_txt = os.path.join('temp',txt_name)
    base_name_video = os.path.basename(opt.video)
    videoreader = VideoReader(opt.video)

    tracking = {}

    max_conf = {}
    target_track = {}
    with open(file_txt) as f:
        lines = f.readlines()
    for line in lines:
        frame_index, track_id, x1,y1,w,h = line.strip().split()[:6]
        conf = line.strip().split()[-1]
        conf = float(conf)
        frame_index = int(frame_index)
        track_id = int(track_id)
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        x2 = x1+w
        y2 = y1+h
        # if max_conf.get(track_id):
        #     if max_conf[track_id] < conf:
        #         max_conf[track_id] = conf
        # else:
        #     max_conf[track_id] = 0 
        max_conf[track_id] = 0 

        target_track[track_id] = {}


        if not tracking.get(frame_index):
            tracking[frame_index] = [{
                'track_id':track_id,
                'cordinates': [x1,y1,x2,y2],
                'conf':conf,
                # 'max_conf': conf
            }]
            
        else:
            tracking[frame_index].append({
                'track_id':track_id,
                'cordinates': [x1,y1,x2,y2],
                'conf':conf
            })
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(opt.save_video,fourcc=fourcc,fps=24,frameSize=(videoreader.width,videoreader.height))

    frame_id = {}
    for ret,frame,frame_index in tqdm(videoreader,total=len(videoreader)):
        frame_info = tracking.get(frame_index)
        frame2 = frame.copy()
        if frame_info:
            for box in frame_info:
                track_id = box['track_id']
                x1,y1,x2,y2 = box['cordinates']
                conf = box['conf']
                # max_conf_track = max_conf[track_id]
                target = target_track.get(track_id)
                # frame_crop_origin = False
                # a = False
                if conf > max_conf[track_id]:
                    max_conf[track_id] = conf
                    frame_id[track_id] = frame[y1:y2,x1:x2,:]
                    
                    frame_crop = cv2.resize(frame_id[track_id],(224,224))
                    inputs = np.transpose(frame_crop,[2,0,1])
                    inputs = inputs.astype('float32')/255.
                    inputs = np.expand_dims(inputs,axis=0)
                    inputs = torch.Tensor(inputs).to('cuda:0')
                    preds_origin = model.predict(inputs)
                    # print(preds)
                    preds = [x.detach().cpu().numpy().argmax(axis=-1).ravel() for x in preds_origin]
                    preds_score = [x.detach().cpu().numpy().max(axis=-1).ravel() for x in preds_origin]
                    for i,(k,v) in enumerate(opt.classes.items()):
                        # print(preds[i])
                        target_track[track_id][k] = {
                            'label':v[preds[i][0]],
                            'score':preds_score[i][0]
                        }
                        
                    # print(target)
                    # exit()
                # frame = cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)), color=(0,0,255), thickness=1)
                label = ''
                for k,v in target.items():
                    temp =v['label']
                    score = v['score']
                    label += f'{temp}-{score:.2f}\n'
                # label += f'conf:{conf}'
                frame2 = box_label(frame2,[x1,y1,x2,y2],label=label)

                # if isinstance(frame_crop_origin,np.ndarray):
                if len(target)>0:
                    
                    frame_crop_origin = cv2.resize(frame_id[track_id],(96,128))
                    try:
                        frame2[int(y1):int(y1)+128,int(x2):int(x2)+96,:] = frame_crop_origin
                    
                    except Exception as e:
                        try:
                            frame2[int(y1):int(y1)+128,int(x1)-96:int(x1),:] = frame_crop_origin
                        except:
                            pass
                        
                        
        writer.write(frame2)





def main():
    opt = get_parser()
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    for k,v in data.items():
        if not hasattr(opt,k):
            setattr(opt,k,v)
    for k,v in cfg.items():
        if not hasattr(opt,k):
            setattr(opt,k,v)
    demo(opt)


if __name__ =='__main__':
    main()