# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np
from K_means import getCluster
import torch.nn as nn
from model import *
from train import *
from ActivationPrune import Conv2dTest,LinearTest
from torch.nn.parameter import Parameter


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
        mask = default_mask.clone() 
        if self.pruning_type == 'conv':
            scps = np.zeros(
                self.clusters_num * default_mask.shape[-1] * default_mask.shape[-1])  
            scps.resize(self.clusters_num, default_mask.shape[-1], default_mask.shape[-1])

            clusters = getCluster(t, self.clusters_num)  

            print(clusters)

            for i in np.arange(0, clusters.shape[0]):  
                for j in np.arange(0, clusters.shape[1]):
                    scp_upgrade(t[i][j], scps[clusters[i][j]])

            scp_binaeryzation(scps, self.cut_num) 
            print(scps)

            for i in np.arange(0, clusters.shape[0]):  
                for j in np.arange(0, clusters.shape[1]):
                    mask[i][j] = torch.from_numpy(scps[clusters[i][j]])

        elif self.pruning_type == 'full':

            scps = np.zeros(self.clusters_num * default_mask.shape[-1])
            scps.resize(self.clusters_num, default_mask.shape[-1])

            clusters = getCluster(t, self.clusters_num)

            print(clusters)

            for i in np.arange(0, clusters.shape[0]):
                scp_upgrade(t[i], scps[int(clusters[i])])

            scp_binaeryzation(scps, self.cut_num)  
            print(scps)

            for i in np.arange(0, clusters.shape[0]):  
                mask[i] = torch.from_numpy(scps[int(clusters[i])])

        return mask

def weightPrune(model_name, ratio, weightPrameter, LinearPrameter, inplace=False):
    def activationWeightPruneOp(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                activationWeightPruneConv = Conv2dTest(
                    ratio,
                    child.in_channels,
                    child.out_channels, child.kernel_size, stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups, bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if child.bias is not None:
                    activationWeightPruneConv.bias = child.bias
                activationWeightPruneConv.weight = Parameter(child.weight)
                module._modules[name] = activationWeightPruneConv


                print(activationWeightPruneConv)
                print(activationWeightPruneConv.weight.shape)
                print('custers_num=6', 'cut_num=', activationWeightPruneConv.weight.shape[-1] * activationWeightPruneConv.weight.shape[-2] / weightPrameter,
                      'pruning_type=conv')
                convPruning = PatternPruningMethod(custers_num=6,
                                                   cut_num=activationWeightPruneConv.weight.shape[-1] * activationWeightPruneConv.weight.shape[
                                                       -2] / weightPrameter,
                                                   pruning_type='conv')
                convPruning.apply(activationWeightPruneConv, 'weight', 6, activationWeightPruneConv.weight.shape[-1] * activationWeightPruneConv.weight.shape[-2] / weightPrameter,
                                  'conv')

                activationWeightPruneConv._forward_pre_hooks

            elif isinstance(child, nn.Linear):
                print(child)
                print(child.weight.shape)
                print('custers_num=4', 'cut_num=', child.weight.shape[-1] / LinearPrameter, 'pruning_type=full')
                fullPruning = PatternPruningMethod(custers_num=8, cut_num=child.weight.shape[-1] / LinearPrameter,
                                                   pruning_type='full')
                fullPruning.apply(child, 'weight', 8, child.weight.shape[-1] / LinearPrameter, 'full')
                child._forward_pre_hooks
            else:
                activationWeightPruneOp(child) 

    if not inplace:
        model = copy.deepcopy(model_name)
    activationWeightPruneOp(model_name) 
    return model_name

def weightPruneUnstructure(model_name,ratio,weightPrameter,LinearPrameter,inplace=False):
    def activationWeightPruneOp(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                prune.random_unstructured(child, name="weight", amount=(1/weightPrameter))
                activationWeightPruneConv = Conv2dTest(
                    ratio,
                    child.in_channels,
                    child.out_channels, child.kernel_size, stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups, bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if child.bias is not None:
                    activationWeightPruneConv.bias = child.bias
                activationWeightPruneConv.weight =  Parameter(child.weight)
                module._modules[name] = activationWeightPruneConv
                child._forward_pre_hooks

            elif isinstance(child, nn.Linear):

                prune.random_unstructured(child, name="weight", amount=(1/LinearPrameter))
                child._forward_pre_hooks
            else:
                activationWeightPruneOp(child)  
    if not inplace:
        model = copy.deepcopy(model_name)
    activationWeightPruneOp( model_name)  
    return model

def getModel(modelName):
    if modelName == 'LeNet':
        return getLeNet()  
    elif modelName == 'AlexNet':
        return getAlexnet()
    elif modelName == 'VGG16':
        return get_vgg16()
    elif modelName == 'SqueezeNet':
        return get_squeezenet()
    elif modelName == 'ResNet':
        return get_resnet18()

def getDataSet(modelName,batchSize,imgSize):
    if modelName == 'VGG16' or modelName == 'AlexNet' or modelName == 'ResNet' or modelName == 'SqueezeNet':
        dataloaders, dataset_sizes = load_cifar10(batch_size=batchSize, pth_path='./data',
                                                  img_size=imgSize)  
    elif modelName == 'LeNet':
        dataloaders, dataset_sizes = load_mnist(batch_size=batchSize, path='./data', img_size=imgSize)

    return dataloaders,dataset_sizes

def weightPruneModelOp(model_name,batch_size,img_size,ratio,pattern,epoch,weightParameter,LinearParameter,quantize,quantize_bit,operation = None):
    net = getModel(model_name)  
    dataloaders, dataset_sizes = getDataSet(model_name,batch_size,img_size)  
    criterion = nn.CrossEntropyLoss()
    if pattern == 'retrain' or pattern == 'train' or pattern == 'DPTrain':
        if pattern == 'retrain':
            if operation == 'activationRetrainAfterWeightRetrain':
                getPth = '../../hdd/hdd_o/pth/' + model_name + '/ratio=0'  + '/Weight' + '/best.pth'
            else:
                getPth = '../../hdd/hdd_o/pth/' + model_name  + '/ratio=' +str(ratio)+ '/Activation' + '/best.pth'  
        else:
            getPth = '../../hdd/hdd_o/pth/' + model_name  + '/ratio=0' + '/Activation' + '/best.pth'
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  
        if operation != 'activationRetrainAfterWeightRetrain':
            net.load_state_dict(torch.load(getPth)) 
            # net = torch.load(getPth)
            if pattern != 'DPTrain':
                weightPrune(net, ratio ,weightParameter,LinearParameter)
            else:
                weightPruneUnstructure(net, ratio ,weightParameter,LinearParameter)
            train_model_jiang(net,dataloaders, dataset_sizes,ratio,'weight', pattern,criterion=criterion, optimizer=optimizer, name=model_name,
                              scheduler=scheduler, num_epochs=epoch, rerun=False)
        else:
            weightPrune(net, ratio, weightParameter, LinearParameter)
            net.load_state_dict(torch.load(getPth))
            train_model_jiang(net, dataloaders, dataset_sizes, ratio, 'weight', pattern, criterion=criterion,
                              optimizer=optimizer, name=model_name,
                              scheduler=scheduler, num_epochs=epoch, rerun=False)

    if pattern == 'test':
        getPth = '../../hdd/hdd_o/pth/' + model_name+ '/ratio=' +str(ratio)+ '/ActivationWeight/' + 'best.pth'
        weightPrune(net, ratio,weightParameter,LinearParameter)
        net.load_state_dict(torch.load(getPth))
        test_model(net, dataloaders, dataset_sizes, criterion=criterion)




    
