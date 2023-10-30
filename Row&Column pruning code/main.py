from ActivationPrune import *
from WeightPrune import weightPruneModelOp
import os
from Op import Op
import torch
import os
import collections
torch.set_printoptions(threshold=5e3,edgeitems=15)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = 'VGG8'  
    batch_size = 2  
    img_size = 32 
    ratio = 1  
    epochA = 1  
    epochAW = 10 
    quantize_w = 8
    quantize_i = 8
    weightParameter = (4/2)
    LinearParameter = (4/3)
    r_pre_layer = {
        'LeNet':[(0.2,0.8),(0.5,0.2),(0.5,0.5)],
        'AlexNet':[(0.6, 0.9), (0.65, 0.4), (0.5, 0.0), (0.55, 0.0), (0.5, 0.0)],
        'VGG16':[(0.2,0.3),(0.4,0.4),(0.3,0.4),(0.6,0.4),(0.4,0.4),(0.6,0.4),(0.6,0.5),(0.4,0.5),(0.6,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4)],
        'VGG8':[(0.2,0.3),(0.6,0.2),(0.55,0.4),(0.6,0.4),(0.55,0.4),(0.6,0.4),(0.55,0.4)],
        'ResNet':[(0.4,0.4) for i in range(20)],
        'NewResNet':[(0.5,0.5) for i in range(20)],
        'ZFNet':[(0.5,0.5) for i in range(5)]
    }
    p_pre_layer = {
        'LeNet':[2,3,1],
        'AlexNet':[2,3,1,1,1],
        'VGG16':[2,1,1,1,1,1,1,1,1,1,1,1,1],
        'VGG8':[2,3,1,1,1,1,1],
        'ResNet':[1 for i in range(20)],
        'NewResNet':[1 for i in range(20)],
        'ZFNet':[1 for i in range(5)]
    }


    operation = 'trainInitialModel'
    lr_rate = 0.0005
    Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)
