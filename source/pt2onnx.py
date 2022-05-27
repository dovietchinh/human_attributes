import torch.onnx
import argparse
import os
import yaml
from models import MobileNetV2,resnet18,resnet34,resnet50,resnet101,resnet152,\
                    resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,\
                    wide_resnet101_2,shufflenet_v2_x0_5,shufflenet_v2_x1_0
from models import model_fn,model_urls

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


def main():
    opt = parse_opt()
    
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    for k,v in cfg.items():
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    opt.classes['age2'] = list(range(76))
    model_config = []
    for k,v in opt.classes.items():
        model_config.append(len(v))
    model = model_fn[opt.model_name](model_config=model_config)
    ckpt_load = torch.load(opt.weights)
    model.load_state_dict(ckpt_load['state_dict'])
    model = model.to('cuda:0')
    input_names = [ "input" ]
    output_names = list(opt.classes.keys())
    dummy_input = torch.randn(1, 3, 224, 224).type(torch.FloatTensor).to('cuda:0')
    
    torch.onnx.export(model, dummy_input, os.path.join(opt.save_dir,opt.model_name + ".onnx"), verbose=True, input_names=input_names, output_names=output_names)

if __name__ =='__main__':
    main()

