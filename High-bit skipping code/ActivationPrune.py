import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
from model import *
from train import *
import json
import random
# from .model import ResNetBasicBlock

from math import sqrt
import copy
from time import time
from Conv2dNew import Execution

import torch.nn.utils.prune as prune
from torch.nn.parameter import Parameter
from quantize import ActivationQuantizer
from quantize import WeightQuantizer
from SRE_com import oneCal

bn_dict = {
    'ResNet':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '6':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '11':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '16':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            }
    },
    'NewResNet':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '6':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '11':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '16':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            }
    },
    'VGG8':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '2':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '3':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '4':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '5':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '6':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '7':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            }
    },
    'AlexNet':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '2':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '3':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '4':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '5':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            }
    },
    'ZFNet':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '2':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '3':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '4':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '5':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            }
    },
    'LeNet':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '2':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '3':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        }
    },
    'VGG16':{
        '1':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '2':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '3':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '4':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '5':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '6':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '7':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '8':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '9':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '10':{
            'bias':None,
            'num_batches_tracked':None,
            'running_mean':None,
            'running_var':None,
            'weight':None
        },
        '11':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '12':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
        '13':{
                'bias':None,
                'num_batches_tracked':None,
                'running_mean':None,
                'running_var':None,
                'weight':None
            },
    }
}

# 定义打印卷积层和全连接层信息的函数
def print_layer(model):
    valueSum, nonzeroSum = 0, 0
    layer_count = 1
    for name, block in model.named_children():
        for layer in block:
            if layer.__class__.__name__ == "ResNetBasicBlock":
                for groupName, group in layer.named_children():
                    for subLayer in group:
                        if isinstance(subLayer, nn.Conv2d):
                            print(layer_count)
                            layer_count += 1
                            print("卷积核的大小为：", subLayer.kernel_size)
                            print("卷积核的stride为：", subLayer.stride)
                            print("卷积层的padding为：", subLayer.padding)
                            print("BN的大小为：", subLayer.in_channels)
                            print('\n')
                            values = subLayer.weight.detach().cpu().numpy()  # numpy格式
                            nonzero = np.count_nonzero(values, axis=None)
                            # print("该卷积层共有{}个权重，非零元素个数为{}\n".format(values.size, nonzero))
                            valueSum += values.size
                            nonzeroSum += nonzero
            else:
                if isinstance(layer, nn.Conv2d):
                    layerName = "卷积层"
                elif isinstance(layer, nn.Linear):
                    layerName = "全连接层"
                else:
                    continue
                # print("{}的尺寸为：".format(layerName), layer.weight.shape)
                values = layer.weight.detach().cpu().numpy()  # numpy格式
                nonzero = np.count_nonzero(values, axis=None)
                # print("该{}共有{}个权重，非零元素个数为{}\n".format(layerName, values.size, nonzero))
                valueSum += values.size
                nonzeroSum += nonzero
    print("总权重个数为{}, 非零元素个数为{}".format(valueSum, nonzeroSum))
# 以上是pyd添加
class Conv2dTest(nn.Conv2d):
    def __init__(self,
                 ratio,
                 layer_count,
                 model_name,
                 quantize,
                 quantize_w,
                 quantize_i,
                 compute_mode,
                 cycle,
                 in_channels,
                 out_channels,
                 kernel_size,
                 r_p_l,
                 r_or_c,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 ):
        super(Conv2dTest, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.ratio = r_p_l
        self.r_or_c = r_or_c
        self.quantize = quantize
        self.quantize_w = quantize_w
        self.quantize_i = quantize_i
        self.layer_count = layer_count
        self.model_name = model_name
        self.compute_mode = compute_mode
        self.cycle = cycle
    def forward(self, input):
        E = Execution(self.ratio,self.r_or_c)
        # E = Execution(self.ratio,self.weight_mask)
        output = E.conv2d(input, self.weight, self.bias, self.stride, self.padding,self.quantize,self.quantize_w,self.quantize_i,self.layer_count,self.model_name,self.compute_mode,self.cycle)
        return output

class LinearTest(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 quantize,
                 quantize_w,
                 quantize_i,
                 bias=True,
                 ):
        super(LinearTest, self).__init__(in_features, out_features, bias)
        self.quantize = quantize
        self.quantize_w = quantize_w
        self.quantize_i = quantize_i
    def forward(self, input):
        # output = F.linear(input, self.weight * self.weight_mask, self.bias)
        self.input = input
        if self.quantize:
            self.activation_quantizer = ActivationQuantizer()
            self.weight_quantizer = WeightQuantizer()
            self.weight_quant = self.quantize_w
            self.input_quant = self.quantize_i
            self.weight.data = self.weight_quantizer(self.weight.data, self.weight_quant,'Linear',None)
            self.input = self.activation_quantizer(self.input,bits=self.input_quant,pattern='Linear')

        output = F.linear(self.input, self.weight, self.bias)
        return output

def prepare(model, ratio,quantize,quantize_w,quantize_i,model_name,compute_mode,cycle,r_pre_layer,p_pre_layer,inplace=False):
    # move intpo prepare
    def addActivationPruneOp(module):
        nonlocal layer_cnt
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                print(layer_cnt)
                activationPruneConv = Conv2dTest(
                    ratio,
                    layer_cnt,
                    model_name,
                    quantize,
                    quantize_w,
                    quantize_i,
                    compute_mode,
                    cycle,
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    r_p_l=r_pre_layer[model_name][layer_cnt],
                    r_or_c=p_pre_layer[model_name][layer_cnt],
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if child.bias is not None:
                    activationPruneConv.bias = Parameter(child.bias)
                activationPruneConv.weight = Parameter(child.weight)
                # activationPruneConv
                #activationPruneConv.weight_mask = Parameter(child.weight_mask)
                module._modules[name] = activationPruneConv
                activationPruneConv._forward_pre_hooks
                layer_cnt += 1
            elif isinstance(child, nn.Linear):
                p_name = str(layer_cnt)
                activationPruneLinear = LinearTest(
                    child.in_features,
                    child.out_features,
                    quantize,
                    quantize_w,
                    quantize_i,
                    bias=(child.bias is not None)
                )
                if child.bias is not None:
                    activationPruneLinear.bias = Parameter(child.bias)
                activationPruneLinear.weight = Parameter(child.weight)
                # activationPruneLinear.weight_mask = Parameter(child.weight_mask)
                module._modules[name] = activationPruneLinear
                # layer_cnt += 1
            elif isinstance(child,nn.BatchNorm2d):
                bn_dict[model_name][str(layer_cnt)]['bias'] = child.bias.data
                bn_dict[model_name][str(layer_cnt)]['num_batches_tracked'] = child.num_batches_tracked
                bn_dict[model_name][str(layer_cnt)]['running_mean'] = child.running_mean.data
                bn_dict[model_name][str(layer_cnt)]['running_var'] = child.running_var.data
                bn_dict[model_name][str(layer_cnt)]['weight'] = child.weight.data
            else:
                addActivationPruneOp(child)  # 这是用来迭代的，Maxpool层的功能是不变的
    layer_cnt = 0
    if not inplace:
        model = copy.deepcopy(model)
    addActivationPruneOp(model)  # 为每一个卷积层添加输入特征图剪枝操作
    np.save('my_file.npy', bn_dict)
    return model

def getModel(modelName):
    if modelName == 'LeNet':
        return getLeNet()  # 加载原始模型框架
    elif modelName == 'AlexNet':
        return getAlexnet()
    elif modelName == 'VGG16':
        return get_thu_vgg16()
    elif modelName == 'ResNet':
        return get_resnet18()
    elif modelName == 'VGG8':
        return getVgg8()
    elif modelName == 'NewResNet':
        return get_newRsenet18()
    elif modelName == 'ZFNet':
        return get_ZFNet()
    # if modelName == 'MobileNet':
    #     return mobilenetv3_large()

def getDataSet(modelName,batchSize,imgSize):
    if modelName == 'VGG16' or modelName == 'AlexNet' or modelName == 'ResNet'  or modelName == 'SqueezeNet' or modelName=='InceptionV3' or modelName =='VGG8' or modelName == 'NewResNet' or modelName == 'ZFNet':
        dataloaders, dataset_sizes = load_cifar10(batch_size=batchSize, pth_path='./data',
                                                  img_size=imgSize)  # 确定数据集
    elif modelName == 'LeNet':
        dataloaders, dataset_sizes = load_mnist(batch_size=batchSize, path='./data', img_size=imgSize)

    return dataloaders,dataset_sizes

def getPruneQuantizeModel(model_name, weight_file_path,pattern,ratio,quantize,quantize_w,quantize_i,compute_mode,cycle,r_pre_layer,p_pre_layer):

    model_orign = torch.load(weight_file_path)
    # print_layer(model_orign)
    activationPruneModel = prepare(model_orign,ratio,quantize,quantize_w,quantize_i,model_name,compute_mode,cycle,r_pre_layer,p_pre_layer)

    return activationPruneModel

def activationPruneModelOp(model_name, batch_size, img_size,pattern,ratio,epoch,quantize,quantize_w,quantize_i,high_bit,compute_mode):
    dataloaders, dataset_sizes = getDataSet(model_name, batch_size, img_size)
    criterion = nn.CrossEntropyLoss()

    weight_file_path = './pth/' + model_name + '/ratio=' + str(ratio) + '/Activation/' + 'best.pth'
    activationPruneModel = getPruneQuantizeModel(model_name, weight_file_path, pattern, ratio,quantize,quantize_w,quantize_i,high_bit,compute_mode)
    test_model(activationPruneModel, dataloaders, dataset_sizes, criterion=criterion)

def highBitPruneModel(model_name, batch_size, img_size,pattern,ratio,epoch,quantize,quantize_w,quantize_i,compute_mode,cycle,r_pre_layer,p_pre_layer):
    dataloaders, dataset_sizes = getDataSet(model_name, batch_size, img_size)
    criterion = nn.CrossEntropyLoss()
    weight_file_path = './pth/' + model_name + '/ratio=' + str(ratio) + '/Activation/' + 'best.pth'
    activationPruneModel = getPruneQuantizeModel(model_name, weight_file_path, pattern, ratio, quantize, quantize_w,quantize_i,compute_mode,cycle,r_pre_layer,p_pre_layer)
    test_model(activationPruneModel, dataloaders, dataset_sizes, criterion=criterion)

