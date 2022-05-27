import torch
from torch2trt import torch2trt
from draft import resnet50

# create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()
model = resnet50(model_config=[3,2,2,2,3]).cuda()
weights = "/u01/Intern/chinhdv/code/multi-task-classification/result/runs_human_attributes_15/best_15.pt"
model.load_state_dict(torch.load(weights)['state_dict'])
# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
model.eval()
# model(x)
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
print(type(model_trt))