import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, class_weights=None, gamma=2,reduction='mean',label_smoothing=0.0):
        super(FocalLoss, self).__init__(class_weights,reduction=reduction)
        self.gamma = gamma
        self.class_weights = class_weights # weight parameter will act as the alpha parameter to balance class weights
        self.label_smoothing = label_smoothing
    def forward(self, input, target):
        # num_samples = target.shape[0]
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.class_weights,label_smoothing=self.label_smoothing,reduce=False)
        pt = torch.exp(-ce_loss)
        # pt = torch.exp(-ce_loss*num_samples)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        focal_loss = focal_loss.sum() if self.reduction=='sum' else focal_loss.mean()
        # print(focal_loss)
    
        return focal_loss



if __name__ == '__main__':
    #test Focal Loss
    a= [[-2.8060e-02, -2.4264e-01,  4.6827e-01],                              
        [ 3.6526e-01, -4.5223e-01,  6.7025e-02],
        [ 6.1475e-02, -4.3193e-02,  3.2006e-01],
        [-3.0019e-01,  5.8267e-02,  3.5362e-01],
        [ 5.4594e-02, -1.5814e-01, -1.2531e-01],
        [-1.5279e-01,  4.0771e-02,  5.7924e-02],
        [ 9.4609e-02, -3.3750e-01,  3.1572e-01],
        [-7.0767e-03, -2.8712e-01, -3.5686e-02],
        [-1.1132e-01, -2.5832e-01,  3.0651e-01],
        [ 1.7716e-01, -1.2108e-01,  4.1604e-01],
        [ 6.1510e-01,  9.3176e-02,  4.3488e-01],
        [-1.2565e-01, -2.9772e-01,  4.1205e-01],
        [-1.9861e-01, -2.6854e-01,  9.0988e-02],
        [ 4.0709e-01, -1.4646e-01,  2.5002e-03],
        [-3.1155e-02, -6.0802e-03, -1.8741e-01],
        [ 6.4203e-02, -3.8000e-01,  1.8890e-01],
        [ 1.8442e-01,  5.3080e-02,  3.7601e-01],
        [-3.8937e-02, -2.2180e-01, -2.1698e-02],
        [ 1.3754e-01,  2.3812e-01,  4.8604e-01],
        [ 1.2683e-02, -1.3138e-01,  3.5791e-01],
        [-2.6964e-01, -1.5229e-02,  1.3930e-01],
        [-4.7184e-02, -1.9978e-01, -2.5241e-01],
        [ 4.5813e-02,  3.8944e-02, -8.7601e-02],
        [-1.9973e-01, -2.1387e-01,  3.2031e-01],
        [-4.6493e-01, -1.6323e-01,  2.2635e-02],
        [ 2.1914e-01, -2.8268e-01,  2.3365e-03],
        [-1.0796e-01, -1.1674e-01, -1.7554e-01],
        [ 5.3110e-02,  1.1269e-01,  1.2455e-01],
        [-8.4007e-02,  2.3946e-01, -3.2124e-02],
        [ 2.6652e-01,  6.3557e-02,  3.4769e-01],
        [-7.4003e-02, -1.1016e-01, -9.6664e-02],
        [ 2.1545e-01, -3.1686e-01,  1.1033e-01],
        [ 2.1012e-01, -8.9429e-02, -8.1031e-02],
        [ 2.9890e-01, -2.4170e-01,  2.0807e-01],
        [-4.0742e-02, -2.2104e-01, -1.1937e-01],
        [ 5.9125e-02, -4.7928e-03, -1.7661e-01],
        [ 6.0740e-02, -3.8878e-01,  9.4714e-02],
        [-2.2414e-01, -1.1310e-01, -1.4218e-01],
        [-1.9134e-02, -1.2216e-01,  4.4841e-01],
        [ 1.3553e-01, -4.3529e-01,  6.4881e-01],
        [-4.7788e-02, -3.2812e-01,  6.4179e-01],
        [ 2.3303e-01, -5.7065e-02, -8.9841e-02],
        [-1.0678e-01, -1.7794e-02,  1.4655e-01],
        [-1.8255e-01, -1.9649e-01, -2.1251e-02],
        [-4.4141e-02,  5.5941e-03, -3.4141e-02],
        [ 3.0988e-02,  2.0235e-01,  1.9260e-01],
        [ 1.0369e-01, -1.7774e-01,  8.9989e-02],
        [-4.6132e-02, -3.9968e-01,  4.8382e-01],
        [-3.4082e-01, -1.2296e-01,  1.9054e-01],
        [ 3.4413e-01, -1.7359e-01,  1.5281e-01],
        [-4.5057e-01, -1.4861e-01,  3.8894e-01],
        [ 1.2297e-01,  3.4884e-02,  5.5864e-01],
        [ 6.1138e-02, -4.4305e-01,  3.1719e-01],
        [-6.9030e-02,  1.8575e-01,  6.2500e-03],
        [-2.1155e-01,  4.5540e-02,  1.1639e-01],
        [-2.4976e-03, -1.1543e-01,  1.0205e-01],
        [-6.8828e-02, -2.7356e-01, -2.3601e-01],
        [ 1.5023e-01,  2.8516e-02, -2.5833e-02],
        [-7.7258e-02, -2.3460e-02, -3.4563e-02],
        [ 3.3189e-02, -2.4988e-01,  4.0488e-01],
        [-1.4701e-01,  2.3855e-01,  1.4947e-01],
        [-3.8474e-01, -3.5501e-01,  1.1752e-01],
        [ 1.1317e-01,  1.3773e-01, -9.2787e-02],
        [ 5.9477e-02, -2.1466e-01, -1.3892e-01],
        [ 7.2400e-02,  2.0187e-02, -2.1513e-02],
        [ 1.6930e-01, -1.6806e-01,  1.1030e-01],
        [ 1.9034e-01,  2.1235e-02, -2.0072e-01],
        [ 8.9754e-02, -1.3844e-01,  1.3712e-02],
        [-2.2787e-01, -6.9607e-02,  5.2977e-01],
        [-1.1514e-01,  2.2345e-01,  2.0058e-01],
        [-1.6442e-01, -1.2583e-01,  1.2733e-01],
        [-1.6159e-01, -2.0424e-01,  2.4416e-01],
        [ 1.0495e-01, -9.7308e-02, -7.9695e-02],
        [ 1.1308e-01,  7.3479e-02, -1.4704e-01],
        [-1.1087e-01, -3.0510e-01, -4.5040e-02],
        [ 1.5290e-01,  5.4195e-02,  2.0064e-01],
        [-1.9265e-01, -8.5754e-02,  1.1485e-01],
        [ 2.4626e-02, -3.0384e-03,  2.4049e-01],
        [-3.3956e-02,  2.3165e-01, -2.0675e-03],
        [ 2.1100e-01, -1.4374e-01,  4.8752e-02],
        [ 3.6237e-01, -5.6608e-01,  2.5577e-01],
        [ 5.8501e-02,  2.2767e-01, -8.9139e-03],
        [-4.1296e-01, -2.9561e-01,  2.2276e-01],
        [-1.4168e-01, -1.9662e-01,  1.7235e-01],
        [-1.2914e-01,  3.3275e-02,  4.1463e-02],
        [ 1.1824e-01,  3.7600e-02,  3.3922e-02],
        [-5.9498e-02, -2.2614e-01,  1.1978e-01],
        [-5.6231e-02,  1.2988e-01,  3.8964e-01],
        [ 1.3765e-01,  1.1971e-01,  2.5671e-01],
        [-5.1963e-02, -2.2122e-02, -4.1463e-03],
        [-2.3749e-01, -2.6040e-01, -3.9409e-02],
        [-4.2045e-02, -3.7054e-02,  4.1743e-01],
        [ 3.3631e-01,  2.4189e-01, -1.5715e-01],
        [ 9.4387e-02, -1.1144e-01,  4.6839e-03],
        [-1.5382e-01, -4.6361e-01,  4.2926e-02],
        [ 1.6819e-01, -5.9337e-02,  3.7059e-01],
        [-2.0828e-01,  1.2180e-01,  1.6248e-02],
        [-1.3994e-01, -8.1151e-03,  1.6234e-01],
        [-3.2051e-01, -8.5562e-02,  4.2977e-01],
        [-1.2662e-01, -4.7462e-01,  1.8440e-02],
        [ 3.0331e-02,  3.5405e-02, -4.6640e-02],
        [ 8.2650e-02, -1.1335e-01,  1.5327e-01],
        [-3.5295e-01, -3.4380e-02, -2.5983e-01],
        [ 2.8515e-01, -1.5784e-01,  2.0264e-01],
        [-1.7704e-01,  7.8075e-02,  1.6604e-01],
        [-1.0198e-01, -2.6845e-01,  3.4353e-01],
        [ 7.8725e-02,  9.1600e-02,  5.2638e-01],
        [-2.7066e-02,  9.9626e-03,  8.1245e-02],
        [ 3.9678e-01, -1.5110e-01,  4.6194e-01],
        [ 7.8305e-02, -7.9402e-02,  1.6519e-01],
        [ 3.8215e-02, -8.3951e-01,  2.7174e-02],
        [-7.4572e-02, -3.2884e-01,  3.1015e-01],
        [ 2.4685e-01, -4.4567e-01, -7.8228e-02],
        [ 1.4415e-01, -2.7336e-01, -1.6168e-01],
        [-2.4863e-01, -2.2159e-01,  1.2193e-01],
        [ 2.1066e-01,  1.3266e-01,  1.8403e-01],
        [-3.1516e-01, -1.0848e-01,  1.2409e-01],
        [-8.7441e-02, -1.4136e-01, -9.2927e-02],
        [ 2.1232e-01,  2.3453e-01, -1.1742e-01],
        [ 2.0944e-01,  1.8655e-01,  1.6708e-01],
        [ 2.3156e-02, -1.9347e-01, -1.5293e-01],
        [-1.8401e-02,  3.0700e-01,  5.0151e-01],
        [-1.5365e-03, -1.5596e-01, -2.3612e-01],
        [ 1.1497e-01,  1.5363e-01,  3.4952e-01],
        [-1.9908e-01,  5.4401e-02,  2.5061e-01],
        [-1.2528e-01,  4.6715e-01, -2.6410e-01],
        [-2.3518e-02, -3.3548e-01, -7.8679e-02],
        [-3.7407e-01, -2.3016e-04,  2.8877e-01]]
    b = [2, 1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 2, 0, 2, 1, 0, 2, 0,
        2, 0, 2, 1, 1, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 2, 2,
        1, 0, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 0, 0, 1, 2, 1, 1, 0, 1, 2, 2, 1, 2,
        2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2,
        1, 2, 2, 1, 1, 0, 1, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2, 1, 0,
        0, 0, 0, 1, 1, 2, 2, 0]
    # a = [[-2.8060e-02, -2.4264e-01,  4.6827e-01],      
    #     [-1.9908e-01,  5.4401e-02,  2.5061e-01],                        
    #     [ 3.6526e-01, -4.5223e-01,  6.7025e-02]]
    # b = [2,1,0]
    a = torch.Tensor(a)
    b=  torch.Tensor(b).type(torch.long)
    # print(a.dtype)
    # print(b.dtype)
    class_weights = torch.Tensor([1,2,3])

    # factor = 0.2

    # # b = (1-factor)*b + factor/b.shape[1]
    # # print(b)

    label_smoothing = 0.2

    # Criterior = torch.nn.CrossEntropyLoss(weight=class_weights,label_smoothing=label_smoothing)
    # loss = Criterior(a,b)
    # print(loss)


    Criterior1 = FocalLoss(gamma=0,class_weights=class_weights, label_smoothing=label_smoothing,reduction='sum')
    Criterior2 = FocalLoss(gamma=0,class_weights=class_weights, label_smoothing=label_smoothing,reduction='mean')
    len_ = a.shape[0]
    # print(len_)
    loss1 = Criterior1(a,b)
    loss2 = Criterior2(a,b)*len_


    print(loss1)
    print(loss2)
    
# a = torch.Tensor([[0.8,0.2]])
# b = torch.Tensor([1]).type(torch.long)
# b = torch.Tensor([[0.1,0.9]])
# Criterior = torch.nn.CrossEntropyLoss()
# Criterior(a,b)