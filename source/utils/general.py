import torch 
import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
LOGGER = logging.getLogger('__main__.'+__name__)

class EarlyStoping:

    def __init__(self, best_epoch=0,best_fitness=0,patience=30,ascending=True):
        self.best_epoch = best_epoch
        self.best_fitness = best_fitness
        self.patience = patience
        self.ascending = ascending
    def __call__(self,epoch,fi):
        if fi >= self.best_fitness and self.ascending:
            self.best_epoch = epoch
            self.best_fitness = fi
        if fi <= self.best_fitness and  not self.ascending: 
            self.best_epoch = epoch
            self.best_fitness = fi
        stop =  (epoch - self.best_epoch) >= self.patience
        if stop:
            LOGGER.info(f' EarlyStopping patience {self.patience} exceeded, stopping training')
        return stop


def visualize(csv,classes_copy,format_index,save_dir,dataset_name='train'):
    LOGGER.info(' loadding data, please wait......')
    classes = classes_copy.copy()
    if isinstance(csv,str):
        csv = pd.read_csv(csv)  

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')   

    for label_name,class_name in classes.items():
        len_ = len(class_name)
        fig = matplotlib.figure.Figure((10,10),dpi=100)
        ax = fig.subplots(1,1)
        x_axes = class_name.copy()
        x_axes.append('unlabeled')
        y_height = []
        for i in range(len_):
            if format_index:
                len_label = (csv[label_name]==i).sum()
            else:
                len_label = (csv[label_name]==class_name[i]).sum()
                
            y_height.append(len_label)
        y_height.append(len(csv[csv[label_name]==-1]))

        cmap = plt.cm.tab10
        colors = cmap(np.arange(len(x_axes)) % cmap.N)

        rec = ax.bar(x_axes,y_height,color=colors)
        autolabel(rec)
        ax.set_title(label_name)
        os.makedirs(os.path.join(save_dir,'visualize'), exist_ok=True)
        fig.savefig(os.path.join(save_dir,'visualize',dataset_name+'_'+label_name+'.png'))

# def log_tensorboard(loss_train,loss_val,)
#     from torch.utils.tensorboard import SummaryWriter

if __name__ =='__main__':
    df = {'path':[1,2,3,4,5,6],
          'age':[0,0,1,2,1,2],
          'gender': [0,1,0,0,-1,1]}
    df = pd.DataFrame(df)
    classes = {
        'age': ['0-18','18-55','55+'],
        'gender': ['female','male'],
    }
    visualize(df,classes,'.')
