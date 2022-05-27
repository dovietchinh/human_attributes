import onnx
import numpy as np
import cv2
import onnxruntime as ort
import pandas as pd
def sofmax(x):
    return np.exp(x)/sum(np.exp(x))

def save_bin():
    df = pd.read_csv('')

def main():
    x = cv2.imread('/u01/Intern/chinhdv/code/multi-task-classification/1_test7.jpg')
    x = cv2.resize(x,(224,224))
    x = x.astype('float32')/255.
    x = np.expand_dims(x,axis=0)
    x = np.transpose(x,[0,3,1,2])
    ort_sess = ort.InferenceSession('/u01/Intern/chinhdv/code/multi-task-classification/mobilenet_v2.onnx')
    output_name = ort_sess.get_outputs()
    name = ['486','487','488','489','490','491','492','493','494','495','496']
    outputs = ort_sess.run(None, {'input.1': x})
    # print(dir(ort_sess))
    for i,j in zip(outputs,name):
        scores = sofmax(i[0])
        max_index,score = np.argmax(scores),np.max(scores)
        print(j,max_index,score)
