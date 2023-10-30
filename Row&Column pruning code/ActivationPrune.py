import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
from model import *
from train import *
import random
# from .model import ResNetBasicBlock

from math import sqrt
import copy
from time import time
from Conv2dNew import Execution

import torch.nn.utils.prune as prune
from K_means import getCluster
import pandas as pd
from torch.nn.parameter import Parameter
from quantize import ActivationQuantizer
from quantize import WeightQuantizer
def scp_upgrade(kernel, old_scp):
    old_scp += np.abs(kernel.cpu().detach().numpy())
    return old_scp
def scp_binaeryzation(scps, C):
    if len(scps.shape) == 3:
        for r in np.arange(0, scps.shape[0]):
            series = pd.Series(scps[r].ravel())
            rank_info = series.rank()
            for i in np.arange(0, scps[r].shape[0]):
                for j in np.arange(0, scps[r].shape[1]):
                    index = i * scps[r].shape[0] + j
                    if (rank_info[index] <= C):
                        scps[r][i][j] = 0
                    else:
                        scps[r][i][j] = 1

    elif len(scps.shape) == 2:
        for r in np.arange(0, scps.shape[0]):
            series = pd.Series(scps[r].ravel())
            rank_info = series.rank()
            for i in np.arange(0, scps[r].shape[0]):
                index = i
                if (rank_info[index] <= C):
                    scps[r][i] = 0
                else:
                    scps[r][i] = 1
class PatternPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, custers_num, cut_num, pruning_type):
        self.clusters_num = custers_num
        self.cut_num = cut_num
        self.pruning_type = pruning_type
        prune.BasePruningMethod.__init__(self)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()  # 复制一个mask大小等于当前层的filter
        if self.pruning_type == 'conv':
            scps = np.zeros(
                self.clusters_num * default_mask.shape[-1] * default_mask.shape[-1])  # 复制num个scp,表示每一个卷积族的pattern
            scps.resize(self.clusters_num, default_mask.shape[-1], default_mask.shape[-1])

            clusters = getCluster(t, self.clusters_num)  # 输入当前层的filter，获得其聚类信息

            print(clusters)

            for i in np.arange(0, clusters.shape[0]):  # 遍历所有kernel,计算所有cluster的scp
                for j in np.arange(0, clusters.shape[1]):
                    scp_upgrade(t[i][j], scps[clusters[i][j]])

            scp_binaeryzation(scps, self.cut_num)  # 根据scp二值化获得真正的pattern
            print(scps)

            for i in np.arange(0, clusters.shape[0]):  # 根据scp和每个kernel的族编号得到最终的mask
                for j in np.arange(0, clusters.shape[1]):
                    mask[i][j] = torch.from_numpy(scps[clusters[i][j]])

        elif self.pruning_type == 'full':

            scps = np.zeros(self.clusters_num * default_mask.shape[-1])
            scps.resize(self.clusters_num, default_mask.shape[-1])

            clusters = getCluster(t, self.clusters_num)

            print(clusters)

            for i in np.arange(0, clusters.shape[0]):
                scp_upgrade(t[i], scps[int(clusters[i])])

            scp_binaeryzation(scps, self.cut_num)  # 根据scp二值化获得真正的pattern
            print(scps)

            for i in np.arange(0, clusters.shape[0]):  # 根据scp和每个kernel的族编号得到最终的mask
                mask[i] = torch.from_numpy(scps[int(clusters[i])])

        return mask
# 定义打印卷积层和全连接层信息的函数
def print_layer(model):
    valueSum, nonzeroSum = 0, 0
    for name, block in model.named_children():
        for layer in block:
            if layer.__class__.__name__ == "ResNetBasicBlock":
                for groupName, group in layer.named_children():
                    for subLayer in group:
                        if isinstance(subLayer, nn.Conv2d):
                            # print("卷积层的尺寸为：", subLayer.weight.shape[0],subLayer.weight.shape[1],subLayer.kernel_size[0],subLayer.stride[0],subLayer.padding[0])
                            values = subLayer.weight.detach().cpu().numpy()  # numpy格式
                            nonzero = np.count_nonzero(values, axis=None)
                            print("该卷积层共有{}个权重，非零元素个数为{}\n".format(values.size, nonzero))
                            valueSum += values.size
                            nonzeroSum += nonzero
            else:
                if isinstance(layer, nn.Conv2d):
                    layerName = "卷积层"
                elif isinstance(layer, nn.Linear):
                    layerName = "全连接层"
                else:
                    continue
                # print("{}的尺寸为：".format(layerName), layer.weight.shape[1], layer.weight.shape[0], layer.kernel_size[0],
                #       layer.stride[0], layer.padding[0])
                print("{}的尺寸为：".format(layerName), layer.weight.shape)
                values = layer.weight.detach().cpu().numpy()  # numpy格式
                nonzero = np.count_nonzero(values, axis=None)
                print("该{}共有{}个权重，非零元素个数为{}\n".format(layerName, values.size, nonzero))
                valueSum += values.size
                nonzeroSum += nonzero
    print("总权重个数为{}, 非零元素个数为{}".format(valueSum, nonzeroSum))
# 以上是pyd添加
class Conv2dTest(nn.Conv2d):
    def __init__(self,
                 ratio,
                 quantize,
                 quantize_w,
                 quantize_i,
                 model_name,
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
        self.model_name = model_name
        self.quantize = quantize
        self.quantize_w = quantize_w
        self.quantize_i = quantize_i
    def forward(self, input):
        E = Execution(self.ratio,self.r_or_c)
        # E = Execution(self.ratio,self.weight_mask)
        output = E.conv2d(input, self.weight, self.bias, self.stride, self.padding,self.quantize,self.quantize_w,self.quantize_i)
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
            self.weight.data = self.weight_quantizer(self.weight.data, bits=self.weight_quant)
            self.input = self.activation_quantizer(self.input,bits=self.input_quant,pattern='Linear')

        print('全连接层激活行',self.input.shape[0]//2,'激活列',self.input.shape[1],'权重行',self.weight.shape[1],"权重列",self.weight.shape[0])
        output = F.linear(self.input, self.weight, self.bias)
        return output

def prepare(model, ratio,quantize,quantize_w,quantize_i,r_pre_layer,p_pre_layer,model_name,inplace=False):
    # move intpo prepare
    def addActivationPruneOp(module):
        nonlocal layer_cnt
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                p_name = str(layer_cnt)
                activationPruneConv = Conv2dTest(
                    ratio,
                    quantize,
                    quantize_w,
                    quantize_i,
                    model_name,
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    r_p_l = r_pre_layer[model_name][layer_cnt],
                    r_or_c = p_pre_layer[model_name][layer_cnt],
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if(layer_cnt == 18):
                    print(model_name)
                if child.bias is not None:
                    activationPruneConv.bias = Parameter(child.bias)
                activationPruneConv.weight = Parameter(child.weight)
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
                layer_cnt += 1
            else:
                addActivationPruneOp(child)  # 这是用来迭代的，Maxpool层的功能是不变的
    layer_cnt = 0
    if not inplace:
        model = copy.deepcopy(model)
    addActivationPruneOp(model)  # 为每一个卷积层添加输入特征图剪枝操作
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


def getDataSet(modelName,batchSize,imgSize):
    if modelName == 'VGG16' or modelName == 'AlexNet' or modelName == 'ResNet'  or modelName == 'SqueezeNet' or modelName=='InceptionV3'or modelName=='VGG8' or modelName == 'NewResNet' or modelName=='ZFNet':
        dataloaders, dataset_sizes = load_cifar10(batch_size=batchSize, pth_path='./data',
                                                  img_size=imgSize)  # 确定数据集
    elif modelName == 'LeNet':
        dataloaders, dataset_sizes = load_mnist(batch_size=batchSize, path='./data', img_size=imgSize)

    return dataloaders,dataset_sizes

def getPruneQuantizeModel(model_name, weight_file_path,pattern,ratio,quantize,quantize_w,quantize_i,r_pre_layer,p_pre_layer):
    if pattern == 'train':
        model_orign = getModel(model_name)  # 测试点1
    else:
        #model_orign.load_state_dict(torch.load(weight_file_path))  # 原始模型框架加载模型信息
        model_orign = torch.load(weight_file_path)

        print_layer(model_orign)
    activationPruneModel = prepare(model_orign,ratio,quantize,quantize_w,quantize_i,r_pre_layer,p_pre_layer,model_name)

    return activationPruneModel

def activationPruneModelOp(model_name, batch_size, img_size,pattern,ratio,epoch,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer):
    dataloaders, dataset_sizes = getDataSet(model_name, batch_size, img_size)
    criterion = nn.CrossEntropyLoss()

    if pattern == 'retrain' or pattern == 'train':
        if quantize == True and ratio == 0 or quantize == False:  # 训练初始的量化模型
            weight_file_path = './pth/' + model_name + '/ratio=0'+ '/Activation' + '/best.pth'
        elif quantize == True and ratio != 0:
            print("翻我了")
            weight_file_path = './pth/' + model_name + '/ratio=0' + '/Quantize' + '/best.pth'
        activationPruneModel = getPruneQuantizeModel(model_name, weight_file_path, pattern, ratio,quantize,quantize_w,quantize_i,r_pre_layer,p_pre_layer)
        optimizer = optim.SGD(activationPruneModel.parameters(), lr=lr_rate, momentum=0.9,weight_decay=0.01)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # 设置学习率下降策略
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch,eta_min=0)
        print("开始训练！")
        train_model_jiang(activationPruneModel, dataloaders, dataset_sizes, ratio, 'activation',quantize,pattern, criterion=criterion,optimizer=optimizer, name=model_name,
                          scheduler=scheduler, num_epochs=epoch, rerun=False)  # 进行模型的训练
    if pattern == 'test':
        weight_file_path = './pth/' + model_name + '/ratio=' + str(ratio) + '/Activation/' + 'best.pth'
        activationPruneModel = getPruneQuantizeModel(model_name, weight_file_path, pattern, ratio,quantize,quantize_w,quantize_i)
        test_model(activationPruneModel, dataloaders, dataset_sizes, criterion=criterion)


