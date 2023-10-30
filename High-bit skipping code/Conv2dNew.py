import math
import numpy as np
import copy
import torch
from statistics import mean
from quantize import ActivationQuantizer
from quantize import WeightQuantizer
import torch.nn as nn
import numpy as np
import time
from imgcol import determine_padding
from imgcol import image_to_column
from imgcol import get_im2col_indices
from imgcol import activationSlidePrune
from SRE_com import *

import torch.nn.functional as F

# from divideTest import shift_bit_Dic

# MaxPooled_kernel_size,MaxPooled_stride,filter_shape,stride,padding,in_channel
Model_Dict = {
    'LeNet': {
        '1': [2, 2, 5, 1, (0, 0),6],
        '2': [2, 2, 5, 1, (0, 0),16],
        'thre': 3
    },
    'AlexNet': {
        '1': [3, 2, 5, 1, (2, 2),96],
        '2': [3, 2, 3, 1, (1, 1),256],
        '3': [0, 0, 3, 1, (1, 1),384],
        '4': [0, 0, 3, 1, (1, 1),384],
        'thre': 5
    },
    'ZFNet': {
        '1': [3, 2, 5, 2, (0, 0),48],
        '2': [3, 2, 3, 1, (1, 1),192],
        '3': [0, 0, 3, 1, (1, 1),192],
        '4': [0, 0, 3, 1, (1, 1),128],
        'thre': 5
    },
    'VGG8':{
        '1': [0, 0, 3, 1, (1, 1),128],
        '2': [2, 2, 3, 1, (1, 1),128],
        '3': [0, 0, 3, 1, (1, 1),256],
        '4': [2, 2, 3, 1, (1, 1),256],
        '5': [0, 0, 3, 1, (1, 1),512],
        '6': [2, 2, 3, 1, (0, 0),512],
        'thre' : 7
    },
    'VGG16': {
        '1': [0, 0, 3, 1, (1, 1),64],
        '2': [2, 2, 3, 1, (1, 1),64],
        '3': [0, 0, 3, 1, (1, 1),128],
        '4': [2, 2, 3, 1, (1, 1),128],
        '5': [0, 0, 3, 1, (1, 1),256],
        '6': [0, 0, 3, 1, (1, 1),256],
        '7': [2, 2, 3, 1, (1, 1),256],
        '8': [0, 0, 3, 1, (1, 1),512],
        '9': [0, 0, 3, 1, (1, 1),512],
        '10': [2, 2, 3, 1, (1, 1),512],
        '11': [0, 0, 3, 1, (1, 1),512],
        '12': [0, 0, 3, 1, (1, 1),512],
        'thre': 13
    },
# MaxPooled_kernel_size,MaxPooled_stride,filter_shape,stride,padding,in_channel
        'ResNet':{
        '1' : [3,2,3,1,(1,1),64],
        '2' : [0,0,3,1,(1,1),64],
        '3' : [0,0,3,1,(1,1),64],
        '4' : [0,0,3,1,(1,1),64],
        '5' : [0,0,1,2,(0,0),64],
        '6' : [0,0,3,2,(1,1),128],
        '7' : [0,0,3,1,(1,1),128],
        '8' : [0,0,3,1,(1,1),128],
        '9' : [0,0,3,1,(1,1),128],
        '10': [0,0,1,2,(0,0),128],
        '11': [0,0,3,2,(1,1),256],
        '12': [0,0,3,1,(1,1),256],
        '13': [0,0,3,1,(1,1),256],
        '14': [0,0,3,1,(1,1),256],
        '15': [0,0,1,2,(0,0),256],
        '16': [0,0,3,2,(1,1),512],
        '17': [0,0,3,1,(1,1),512],
        '18': [0,0,3,1,(1,1),512],
        '19': [0,0,3,1,(1,1),512],
        'thre':20
    },
# MaxPooled_kernel_size,MaxPooled_stride,filter_shape,stride,padding,out_channel
        'NewResNet':{
        '1' : [0,0,3,1,(1,1),64],
        '2' : [0,0,3,1,(1,1),64],
        '3' : [0,0,3,1,(1,1),64],
        '4' : [0,0,3,1,(1,1),64],
        '5' : [0,0,1,2,(0,0),64],
        '6' : [0,0,3,2,(1,1),128],
        '7' : [0,0,3,1,(1,1),128],
        '8' : [0,0,3,1,(1,1),128],
        '9' : [0,0,3,1,(1,1),128],
        '10': [0,0,1,2,(0,0),128],
        '11': [0,0,3,2,(1,1),256],
        '12': [0,0,3,1,(1,1),256],
        '13': [0,0,3,1,(1,1),256],
        '14': [0,0,3,1,(1,1),256],
        '15': [0,0,1,2,(0,0),256],
        '16': [0,0,3,2,(1,1),512],
        '17': [0,0,3,1,(1,1),512],
        '18': [0,0,3,1,(1,1),512],
        '19': [0,0,3,1,(1,1),512],
        'thre':20
        }
}

shift_bit_Dic = {
    'LeNet': [[[0 for i in range(3)], [0 for i in range(3)], [0 for i in range(3)]]],
    'VGG8': [
        # [[0 for i in range(7)], [0 for i in range(7)], [0 for i in range(7)]],
        [[2,3,2,3,2,4,4], [0 for i in range(7)], [0 for i in range(7)]],
    ],
    'VGG16': [
        # [[0 for i in range(13)],[0 for i in range(13)],[0 for i in range(13)]],
        [[2,2,2,3,2,2,5,2,2,3,3,5,5], [0 for i in range(13)],[0 for i in range(13)]],
    ],
    'AlexNet': [
        # [[0,0,0,0,0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        [[4,3,3,4,4], [0,0,0,0,0], [0,0,0,0,0]]
    ],
    'ResNet': [
        # [[0 for i in range(20)], [0 for i in range(20)], [0 for i in range(20)]],
        [[3,3,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2], [0 for i in range(20)], [0 for i in range(20)]],
    ],
    'NewResNet':[
        # [[0 for i in range(20)], [0 for i in range(20)], [0 for i in range(20)]],
        [[3,3,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2], [0 for i in range(20)], [0 for i in range(20)]]
    ],
    'ZFNet':[
        # [[0 for i in range(5)], [0 for i in range(5)], [0 for i in range(5)]],
        [[3,2,2,3,3], [0 for i in range(5)], [0 for i in range(5)]]
        ]
}

r_pre_layer = {
        'LeNet': [(0.2, 0.8), (0.5, 0.2), (0.5, 0.5)],
        'AlexNet': [(0.6, 0.9), (0.65, 0.4), (0.5, 0.0), (0.55, 0.0), (0.5, 0.0)],
        'VGG16': [(0.2,0.3),(0.4,0.4),(0.3,0.4),(0.6,0.4),(0.4,0.4),(0.6,0.4),(0.6,0.5),(0.4,0.5),(0.6,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4)],
        'VGG8':  [(0.2, 0.3), (0.6, 0.2), (0.55, 0.4), (0.6, 0.4), (0.55, 0.4), (0.6, 0.4), (0.55, 0.4)],
        'ResNet': [(0.4, 0.4) for i in range(20)],
        'NewResNet': [(0.5,0.5) for i in range(20)],
        'ZFNet': [(0.35, 0.35), (0.5, 0.3), (0.5, 0), (0.5, 0), (0.5, 0)]
}
#
# r_pre_layer = {
#         'LeNet': [(0,0) for i in range(3)],
#         'AlexNet': [(0,0) for i in range(5)],
#         'VGG16': [(0,0) for i in range(13)],
#         'VGG8':  [(0,0) for i in range(7)],
#         'ResNet': [(0.4, 0.4) for i in range(20)],
#         'NewResNet': [(0,0) for i in range(20)],
#         'ZFNet': [(0, 0) for i in range(5)]
# }
p_pre_layer = {
    'LeNet': [2, 3, 1],
    'AlexNet': [2, 3, 1, 1, 1],
    'VGG16': [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'VGG8': [2, 3, 1, 1, 1, 1, 1],
    'ResNet': [1 for i in range(20)],
    'NewResNet': [1 for i in range(20)],
    'ZFNet':[2,1,1,1,1]
}


resnet_count = 0
resnet_Dict={
    'need_reserve':[0,2,5,7,10,12,15,17],
    'need_read_reserve':[2,4,7,9,12,14,17,19],
    'predict_twice':[4,9,14],  
    'without_predict':[5,10,15],  
    'need_relu':[0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19],  
    'need_bn':[0,5,10,15]
}
resnet_reserve_dict={}



class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()

class Execution(Layer):
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """

    def __init__(self, ratio,r_or_c,weight_mask=None):
        self.ratio = ratio
        self.r_or_c = r_or_c

        # self.weight_mask = weight_mask
        pass

    def conv2d(self, input, weight, bias, stride, padding, quantize, quantize_w, quantize_i, layer_count, model_name,
               compute_mode, cycle):
        global LeNet_Dict
        self.input = input
        # self.weight = weight * self.weight_mask
        self.layer_count = layer_count
        self.model_name = model_name
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.compute_mode = compute_mode
        self.cycle = cycle
        self.n_filters = self.weight .shape[0]  
        self.filter_shape = (self.weight.shape[2], self.weight.shape[3])
        self.input_shape = [self.input.shape[1], self.input.shape[2], self.input.shape[3]]
        self.trainable = False
        batch_size, channels, height, width =self.input.shape
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        if self.model_name != 'ResNet' and self.model_name != 'NewResNet':
            if quantize == True:
                # p = self.input.to('cpu').numpy()
                # q = self.weight.to('cpu').numpy()
                self.weight_quant = quantize_w  
                self.input_quant = quantize_i  
                self.weight = self.weight.reshape((self.n_filters, -1)) 

                self.input = image_to_column(self.input, self.filter_shape, stride=self.stride,
                                             output_shape=self.padding)  
                if (self.compute_mode == 0 or self.compute_mode == 2):
                    if self.layer_count == 0:  
                        min_val, max_val = self.input.min(), self.input.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                        self.input = torch.round(self.input / scale_i)  
                        self.input = self.activationSlidePrune(self.input, self.ratio,self.r_or_c)
                        pre_scale_i = scale_i
                        pre_input_8 = self.input
                        self.input = self.input * scale_i  
                    else:
                        min_val, max_val = self.input.min(), self.input.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                        self.input = torch.round(self.input / scale_i)
                        pruneTensor = Model_Dict[self.model_name][
                            str(layer_count + Model_Dict[model_name]['thre'] - 1)] 
                        if(pruneTensor[0] == 1):
                            self.input[(pruneTensor[1] == 1),] = 0
                        elif(pruneTensor[0] == 2):
                            self.input[:, (pruneTensor[1] == 1)] = 0
                        else:
                            self.input[(pruneTensor[1] == 1),] = 0
                            self.input[:, (pruneTensor[2] == 1)] = 0
                        pre_scale_i = scale_i
                        pre_input_8 = self.input
                        self.input = self.input * scale_i
                if (self.compute_mode == 1):
                    min_val, max_val = self.input.min(), self.input.max()
                    min_val, max_val = min_val.item(), max_val.item()
                    scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                    self.input = torch.round(self.input / scale_i)
                    self.input = self.activationSlidePrune(self.input, self.ratio,self.r_or_c)
                    pre_scale_i = scale_i
                    pre_input_8 = self.input
                    self.input = self.input * scale_i
                a = self.output_shape() + (batch_size,)
                output = comNew(self.input, self.weight, self.bias, self.weight_quant, self.layer_count, batch_size, a,
                                self.ratio, model_name, shift_bit_Dic[model_name][self.cycle][0][self.layer_count],
                                shift_bit_Dic[model_name][self.cycle][1][self.layer_count],
                                shift_bit_Dic[model_name][self.cycle][2][self.layer_count]
                                , self.compute_mode, pre_input_8,pre_scale_i,self.r_or_c)
                output = output.reshape(self.output_shape() + (batch_size,))
                output = output.permute(3, 0, 1, 2)
                return output
            if quantize == False:
                self.X_col = image_to_column(self.input, self.filter_shape, stride=self.stride, output_shape=self.padding)
                # Turn weights into column shape
                if self.ratio != 0:
                    self.X_col = self.activationSlidePrune(self.X_col, self.ratio)
                self.W_col = self.weight.reshape((self.n_filters, -1))
                # Calculate output
                if self.bias is not None:
                    output = torch.einsum('ij,jk->ik', self.W_col, self.X_col) + (torch.unsqueeze(self.bias, 1))
                else:
                    output = torch.einsum('ij,jk->ik', self.W_col, self.X_col)
                output = output.reshape(self.output_shape() + (batch_size,))
                return output.permute(3, 0, 1, 2)
        else:
            if quantize == True:
                # p = self.input.to('cpu').numpy()
                # q = self.weight.to('cpu').numpy()
                self.weight_quant = quantize_w  
                self.input_quant = quantize_i  
                self.weight = self.weight.reshape((self.n_filters, -1))  
                input_original = self.input
                self.input = image_to_column(self.input, self.filter_shape, stride=self.stride,
                                             output_shape=self.padding)  
                if (self.compute_mode == 0 or self.compute_mode == 2):
                    if self.layer_count == 0:  
                        min_val, max_val = self.input.min(), self.input.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                        self.input = torch.round(self.input / scale_i)
                        self.input = self.activationSlidePrune(self.input, self.ratio,self.r_or_c)
                        pre_scale_i = scale_i
                        pre_input_8 = self.input
                        self.input = self.input * scale_i
                    else:
                        min_val, max_val = self.input.min(), self.input.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                        self.input = torch.round(self.input / scale_i)
                        pruneTensor = Model_Dict[self.model_name][
                            str(layer_count + Model_Dict[model_name]['thre'] - 1)]  

                        if (pruneTensor[0] == 1):
                            self.input[(pruneTensor[1] == 1),] = 0
                        elif (pruneTensor[0] == 2):
                            self.input[:, (pruneTensor[1] == 1)] = 0
                        else:
                            self.input[(pruneTensor[1] == 1),] = 0
                            self.input[:, (pruneTensor[2] == 1)] = 0
                        pre_scale_i = scale_i
                        pre_input_8 = self.input
                        self.input = self.input * scale_i



                if (self.compute_mode == 1):
                    min_val, max_val = self.input.min(), self.input.max()
                    min_val, max_val = min_val.item(), max_val.item()
                    scale_i = (max_val - min_val) / (2 ** self.input_quant - 1)
                    self.input = torch.round(self.input / scale_i)
                    self.input = self.activationSlidePrune(self.input, self.ratio)
                    pre_scale_i = scale_i
                    pre_input_8 = self.input
                    self.input = self.input * scale_i
                a = self.output_shape() + (batch_size,)

                output = comNew(self.input, self.weight, self.bias, self.weight_quant, self.layer_count, batch_size, a,
                                self.ratio, model_name, shift_bit_Dic[model_name][self.cycle][0][self.layer_count],
                                shift_bit_Dic[model_name][self.cycle][1][self.layer_count],
                                shift_bit_Dic[model_name][self.cycle][2][self.layer_count]
                                , self.compute_mode, pre_input_8, pre_scale_i, self.r_or_c)
                output = output.reshape(self.output_shape() + (batch_size,))
                output = output.permute(3, 0, 1, 2)

                return output
            if quantize == False:
                self.X_col = image_to_column(self.input, self.filter_shape, stride=self.stride, output_shape=self.padding)
                # Turn weights into column shape
                if self.ratio != 0:
                    self.X_col = self.activationSlidePrune(self.X_col, self.ratio)
                self.W_col = self.weight.reshape((self.n_filters, -1))
                # Calculate output
                if self.bias is not None:
                    output = torch.einsum('ij,jk->ik', self.W_col, self.X_col) + (torch.unsqueeze(self.bias, 1))
                else:
                    output = torch.einsum('ij,jk->ik', self.W_col, self.X_col)
                output = output.reshape(self.output_shape() + (batch_size,))
                return output.permute(3, 0, 1, 2)

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride[0] + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride[0] + 1
        return self.n_filters, int(output_height), int(output_width)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def compressionRateStatistics(self, input, andSum, compareRatio):
        pruneNumber = 0
        zerosNumber = 0
        for i in range(input.shape[0]):
            if andSum[i] == 0:
                zerosNumber += 1
            if andSum[i] != 0 and andSum[i] <= compareRatio:
                pruneNumber += 1
        print('pruneNumberRatio=', pruneNumber / (input.shape[0]))
        print('zerosNumberRatio=', zerosNumber / (input.shape[0]))

    def accuracyTest(self, andSum):
        for i in range(len(andSum)):
            print(i, andSum[i])

    def activationSlidePrune(self, input, ratio,r_or_c, pattern='Train'):
        matrixOne = torch.ones(input.shape, device='cuda:0')  
        # x = copy.deepcopy(input)
        x = torch.clone(torch.detach(input))
        andOp = torch.logical_and(matrixOne, x) 

        if r_or_c == 1:
            andSum_row = torch.sum(andOp,dim=1)  
            list_queue = torch.sort(andSum_row)
            num = torch.floor(torch.tensor(len(list_queue.values)*ratio[0]))
            r = list_queue.values[int(num)]
            pruneTensor_row = torch.zeros_like(andSum_row)
            if r == 0:
                pruneTensor_row[(andSum_row <= r),] = 1
            else:
                pruneTensor_row[(andSum_row < r),] = 1
            if pattern == 'test':
                return (1,pruneTensor_row)
            else:
                input[(andSum_row < r),] = 0
                return input

        elif r_or_c == 2:
            andSum_column = torch.sum(andOp, dim=0)  
            list_queue = torch.sort(andSum_column)
            num = torch.floor(torch.tensor(len(list_queue.values) * ratio[1]))
            r = list_queue.values[int(num)]
            pruneTensor_column = torch.zeros_like(andSum_column)
            if(r == 0):
                pruneTensor_column[(andSum_column <= r),] = 1
            else:
                pruneTensor_column[(andSum_column < r),] = 1
            if pattern == 'test':
                return (2,pruneTensor_column)
            else:
                input[:,(andSum_column < r)] = 0
                return input

        else:
            andSum_row = torch.sum(andOp, dim=1)
            andSum_column = torch.sum(andOp, dim=0)

            list_queue_r1 = torch.sort(andSum_row)
            list_queue_r2 = torch.sort(andSum_column)

            num_r1 = torch.floor(torch.tensor(len(list_queue_r1.values) * ratio[0]))
            num_r2 = torch.floor(torch.tensor(len(list_queue_r2.values) * ratio[1]))

            r1 = list_queue_r1.values[int(num_r1)]
            r2 = list_queue_r2.values[int(num_r2)]

            pruneTensor_row = torch.zeros_like(andSum_row)
            if r1 == 0:
                pruneTensor_row[(andSum_row <= r1),] = 1
            else:
                pruneTensor_row[(andSum_row < r1),] = 1

            pruneTensor_column = torch.zeros_like(andSum_column)
            if r2 == 0:
                pruneTensor_column[(andSum_column <= r2),] = 1
            else:
                pruneTensor_column[(andSum_column < r2),] = 1
            if pattern == 'test':
                return (3,pruneTensor_row,pruneTensor_column)
            else:
                input[(andSum_row < r1),] = 0
                input[:, (andSum_column < r2)] = 0
                return input

def comNew(input, weight, bias, bits, layer_count, batch_size, a, ratio, model_name, shift_bit_input, shift_bit_weight,
           shift_bit_bias, compute_mode,pre_input_8,pre_scale_i,r_or_c,input_original=None):
    if model_name != 'ResNet' and model_name != 'NewResNet':
        global resnet_count
        input_8 = pre_input_8
        scale_i = pre_scale_i
        min_val, max_val = weight.min(), weight.max()
        min_val, max_val = min_val.item(), max_val.item()
        scale_w = (max_val - min_val) / (2 ** bits - 1)
        weight_8 = torch.round(weight / scale_w)  
        if bias is not None:
            min_val, max_val = bias.min(), bias.max()
            min_val, max_val = min_val.item(), max_val.item()
            scale_b = (max_val - min_val) / (2 ** bits - 1)
            bias_8 = torch.round(bias / scale_b)

        input_8_np = input_8.to('cpu').numpy()
        weight_8_np = weight_8.to('cpu').numpy()
        if bias is not None:
            bias_8_np = bias_8.to('cpu').numpy()

        input_4_np = np.trunc((np.trunc(input_8_np / int(2 ** shift_bit_input))) * int(2 ** shift_bit_input))
        weight_4_np = np.trunc((np.trunc(weight_8_np / int(2 ** shift_bit_weight))) * int(2 ** shift_bit_weight))
        if bias is not None:
            bias_4_np = np.trunc((np.trunc(bias_8_np / int(2 ** shift_bit_bias))) * int(2 ** shift_bit_bias))

        input_4 = torch.from_numpy(input_4_np).cuda()
        weight_4 = torch.from_numpy(weight_4_np).cuda()
        if bias is not None:
            bias_4 = torch.from_numpy(bias_4_np).cuda()

        if bias is not None:
            o_4 = torch.einsum('ij,jk->ik', weight_4 * scale_w, input_4 * scale_i) + (torch.unsqueeze(bias*scale_b, 1))
        else:
            o_4 = torch.einsum('ij,jk->ik', weight_4 * scale_w, input_4 * scale_i)
        if (compute_mode == 0 or compute_mode == 2):  
            if layer_count == Model_Dict[model_name]['thre'] - 1: 
                temp_out = copy.deepcopy(o_4)
                temp_out = temp_out.reshape(a)
                temp_out = temp_out.permute(3, 0, 1, 2)  
                if model_name == 'VGG8':
                    BN_size = 1024
                elif model_name == 'VGG16':
                    BN_size = 512
                elif model_name == 'AlexNet':
                    BN_size = 256
                elif model_name == 'ZFNet':
                    BN_size = 128
                bn = nn.BatchNorm2d(BN_size)
                load_dict = np.load('my_file.npy', allow_pickle=True).item()
                bn.bias.data = load_dict[str(model_name)][str(layer_count+1)]['bias']
                bn.num_batches_tracked.data = load_dict[str(model_name)][str(layer_count+1)]['num_batches_tracked']
                bn.running_mean.data = load_dict[str(model_name)][str(layer_count+1)]['running_mean']
                bn.running_var.data = load_dict[str(model_name)][str(layer_count+1)]['running_var']
                bn.weight.data = load_dict[str(model_name)][str(layer_count+1)]['weight']
                bn.training = False
                temp_out = bn(temp_out)
                shape_cal = [temp_out.shape[0],temp_out.shape[1],temp_out.shape[2],temp_out.shape[3]]  

                relu = nn.ReLU()
                temp_out = relu(temp_out)

                if model_name == 'ZFNet' or model_name == 'AlexNet':
                    maxpool = nn.MaxPool2d(3, 2)
                else:
                    maxpool = nn.MaxPool2d(2, 2)
                temp_out = maxpool(temp_out)
                logic_temp = torch.ones_like(temp_out)
                need_cal = torch.logical_and(logic_temp, temp_out)  

            if layer_count <= Model_Dict[model_name]['thre'] - 2:  
                temp_out = copy.deepcopy(o_4)
                temp_out = temp_out.reshape(a)
                temp_out = temp_out.permute(3, 0, 1, 2)  
                BN_size = Model_Dict[model_name][str(layer_count + 1)][5]
                bn = nn.BatchNorm2d(BN_size)
                load_dict = np.load('my_file.npy', allow_pickle=True).item()
                bn.bias.data = load_dict[str(model_name)][str(layer_count + 1)]['bias']
                bn.num_batches_tracked.data = load_dict[str(model_name)][str(layer_count + 1)]['num_batches_tracked']
                bn.running_mean.data = load_dict[str(model_name)][str(layer_count + 1)]['running_mean']
                bn.running_var.data = load_dict[str(model_name)][str(layer_count + 1)]['running_var']
                bn.weight.data = load_dict[str(model_name)][str(layer_count + 1)]['weight']
                bn.training = False
                temp_out = bn(temp_out)
                shape_cal = [temp_out.shape[0],temp_out.shape[1],temp_out.shape[2],temp_out.shape[3]]  
                relu = nn.ReLU()
                temp_out = relu(temp_out)
                if Model_Dict[model_name][str(layer_count + 1)][0] != 0:
                    kernel_size_maxpool = (
                    Model_Dict[model_name][str(layer_count + 1)][0], Model_Dict[model_name][str(layer_count + 1)][0])
                    stride_maxpool = Model_Dict[model_name][str(layer_count + 1)][1], \
                                     Model_Dict[model_name][str(layer_count + 1)][1]
                    padding_maxpool = (0, 0)
                    if model_name == 'ZFNet':
                        maxpool = nn.MaxPool2d(kernel_size_maxpool, stride_maxpool,padding=1)
                    else:
                        maxpool = nn.MaxPool2d(kernel_size_maxpool,stride_maxpool)
                    temp_out = maxpool(temp_out)

                logic_temp = torch.ones_like(temp_out)
                need_cal = torch.logical_and(logic_temp, temp_out) 
        
                min_val, max_val = temp_out.min(), temp_out.max()
                min_val, max_val = min_val.item(), max_val.item()
                scale_temp = (max_val - min_val) / (2 ** bits - 1)
                temp_out = torch.round(temp_out / scale_temp)
                temp_out = image_to_column(temp_out, (
                Model_Dict[model_name][str(layer_count + 1)][2], Model_Dict[model_name][str(layer_count + 1)][2]),
                                         stride=(Model_Dict[model_name][str(layer_count + 1)][3],
                                                 Model_Dict[model_name][str(layer_count + 1)][3]),
                                         output_shape=Model_Dict[model_name][str(layer_count + 1)][4])

                prunetensor = activationSlidePrune(temp_out, r_pre_layer[model_name][layer_count+1],p_pre_layer[model_name][layer_count+1],pattern='test')
                Model_Dict[model_name][str(layer_count + Model_Dict[model_name]['thre'])] = prunetensor

            if (compute_mode == 0):
                if bias is not None:
                    o_8 = torch.einsum('ij,jk->ik', weight_8*scale_w, input) + (
                        torch.unsqueeze(bias, 1))
                else:
                    o_8 = torch.einsum('ij,jk->ik', weight_8*scale_w, input)
                return o_8

        if (compute_mode == 1 or compute_mode == 2):
            if layer_count != Model_Dict[model_name]['thre'] - 1: 
                for i in range(o_4.shape[0]):
                    w1 = weight_8[i].view(1, len(weight_8[i]))
                    if bias is not None:
                        o_8_vv = torch.einsum('ij,jk->ik', w1*scale_w, input) + bias_8[i]*scale_b
                    else:
                        o_8_vv = torch.einsum('ij,jk->ik', w1*scale_w , input)
                    w1_4 = weight_4[i].view(1, len(weight_4[i]))
                    if bias is not None:
                        o_4_vv = torch.einsum('ij,jk->ik', w1_4*scale_w, input_4*scale_i) + bias_4[i]*scale_b
                    else:
                        o_4_vv = torch.einsum('ij,jk->ik', w1_4*scale_w, input_4*scale_i)
                    o_8_vv[:, (o_4[i] <= 0)] = 0
                    o_4_vv[:, (o_4[i] > 0)] = 0  
                    o = o_8_vv + o_4_vv
                    o_4[i] = o
                return o_4
            else:
                if bias is not None:
                    o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input_8 * scale_i) + (
                        torch.unsqueeze(bias * scale_b, 1))
                else:
                    o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input_8 * scale_i)
                return o_8
    else:
        global LeNet_Dict
        input_8 = pre_input_8
        scale_i = pre_scale_i
        min_val, max_val = weight.min(), weight.max()
        min_val, max_val = min_val.item(), max_val.item()
        scale_w = (max_val - min_val) / (2 ** bits - 1)
        weight_8 = torch.round(weight / scale_w)  
        if bias is not None:
            min_val, max_val = bias.min(), bias.max()
            min_val, max_val = min_val.item(), max_val.item()
            scale_b = (max_val - min_val) / (2 ** bits - 1)
            bias_8 = torch.round(bias / scale_b)

        input_8_np = input_8.to('cpu').numpy()
        weight_8_np = weight_8.to('cpu').numpy()
        if bias is not None:
            bias_8_np = bias_8.to('cpu').numpy()

        input_4_np = np.trunc((np.trunc(input_8_np / int(2 ** shift_bit_input))) * int(2 ** shift_bit_input))
        weight_4_np = np.trunc((np.trunc(weight_8_np / int(2 ** shift_bit_weight))) * int(2 ** shift_bit_weight))
        if bias is not None:
            bias_4_np = np.trunc((np.trunc(bias_8_np / int(2 ** shift_bit_bias))) * int(2 ** shift_bit_bias))
        input_4 = torch.from_numpy(input_4_np).cuda()
        weight_4 = torch.from_numpy(weight_4_np).cuda()

        if bias is not None:
            bias_4 = torch.from_numpy(bias_4_np).cuda()

        if bias is not None:
            o_4 = torch.einsum('ij,jk->ik', weight_4 * scale_w, input_4 * scale_i) + (
                torch.unsqueeze(bias_4 * scale_b, 1))
        else:
            o_4 = torch.einsum('ij,jk->ik', weight_4 * scale_w, input_4 * scale_i)

        if bias is not None:
            o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input) + (
                torch.unsqueeze(bias_8* scale_b, 1))
        else:
            o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input)


        o_8_t = copy.deepcopy(o_8)
        o_8_t = o_8_t.reshape(a)
        o_8_t = o_8_t.permute(3, 0, 1, 2)

        if (compute_mode == 0 or compute_mode == 2):
            if layer_count == Model_Dict[model_name]['thre'] - 1:  
                temp_out = copy.deepcopy(o_4)
                temp_out = temp_out.reshape(a)
                temp_out = temp_out.permute(3, 0, 1, 2) 
                shape_cal = [temp_out.shape[0], temp_out.shape[1], temp_out.shape[2],
                             temp_out.shape[3]]  
                relu = nn.ReLU()
                temp_out = relu(temp_out)
                maxpool = nn.MaxPool2d(3, 2)
                temp_out = maxpool(temp_out)

                logic_temp = torch.ones_like(temp_out)
                need_cal = torch.logical_and(logic_temp, temp_out)  
            if layer_count <= Model_Dict[model_name]['thre'] - 2:
                if layer_count not in resnet_Dict['without_predict']:
                    if layer_count in resnet_Dict['need_read_reserve']:
                        readActivation = resnet_reserve_dict[resnet_count]
                        readActivation = readActivation.to('cuda:0')
                        o_8_t += readActivation
                        resnet_count += 1
                    temp_out = copy.deepcopy(o_4)
                    temp_out = temp_out.reshape(a)
                    temp_out = temp_out.permute(3, 0, 1, 2)
                    shape_cal = [temp_out.shape[0], temp_out.shape[1], temp_out.shape[2],
                                 temp_out.shape[3]]
                    if layer_count in resnet_Dict['need_read_reserve']:
                        temp_out += readActivation
                    if (layer_count == 0):  
                        BN_size = Model_Dict[model_name][str(layer_count + 1)][5]
                        bn = nn.BatchNorm2d(BN_size)
                        load_dict = np.load('my_file.npy',allow_pickle=True).item()
                        bn.bias.data =                load_dict[str(model_name)][str(layer_count+1)]['bias']
                        bn.num_batches_tracked.data = load_dict[str(model_name)][str(layer_count + 1)]['num_batches_tracked']
                        bn.running_mean.data        = load_dict[str(model_name)][str(layer_count + 1)]['running_mean']
                        bn.running_var.data         = load_dict[str(model_name)][str(layer_count + 1)]['running_var']
                        bn.weight.data              = load_dict[str(model_name)][str(layer_count + 1)]['weight']
                        bn.training = False
                        temp_out = bn(temp_out)
                        o_8_t = bn(o_8_t)
                    if (layer_count in resnet_Dict['need_relu']):
                        relu = nn.ReLU()
                        temp_out = relu(temp_out)
                        if(layer_count in resnet_Dict['need_reserve']):
                            o_8_t = relu(o_8_t)

                    if(layer_count == 0 and model_name == 'ResNet'):  
                        if Model_Dict[model_name][str(layer_count + 1)][0] != 0:
                            kernel_size_maxpool = (
                                Model_Dict[model_name][str(layer_count + 1)][0],
                                Model_Dict[model_name][str(layer_count + 1)][0])
                            stride_maxpool = Model_Dict[model_name][str(layer_count + 1)][1], \
                                             Model_Dict[model_name][str(layer_count + 1)][1]
                            padding_maxpool = (1, 1)
                            maxpool = nn.MaxPool2d(kernel_size_maxpool, stride_maxpool,padding_maxpool)
                            temp_out = maxpool(temp_out)
                            o_8_t = maxpool(o_8_t)
                    logic_temp = torch.ones_like(temp_out)
                    need_cal = torch.logical_and(logic_temp, temp_out) 

                    if layer_count in resnet_Dict['predict_twice']:
                        temp_out_2 = copy.deepcopy(temp_out)
                        min_val, max_val = temp_out.min(), temp_out.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_temp = (max_val - min_val) / (2 ** bits - 1)
                        temp_out = torch.round(temp_out / scale_temp)
                        temp_out = image_to_column(temp_out, (
                            Model_Dict[model_name][str(layer_count + 1)][2],
                            Model_Dict[model_name][str(layer_count + 1)][2]),
                                                   stride=(Model_Dict[model_name][str(layer_count + 1)][3],
                                                           Model_Dict[model_name][str(layer_count + 1)][3]),
                                                   output_shape=Model_Dict[model_name][str(layer_count + 1)][4])
                        prunetensor = activationSlidePrune(temp_out, r_pre_layer[model_name][layer_count + 1],
                                                           p_pre_layer[model_name][layer_count + 1], pattern='test')
                        Model_Dict[model_name][str(layer_count + Model_Dict[model_name]['thre'])] = prunetensor

                        temp_out = copy.deepcopy(temp_out_2)
                        min_val, max_val = temp_out.min(), temp_out.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_temp = (max_val - min_val) / (2 ** bits - 1)
                        temp_out = torch.round(temp_out / scale_temp)
                        temp_out = image_to_column(temp_out, (
                            Model_Dict[model_name][str(layer_count + 2)][2],
                            Model_Dict[model_name][str(layer_count + 2)][2]),
                                                   stride=(Model_Dict[model_name][str(layer_count + 2)][3],
                                                           Model_Dict[model_name][str(layer_count + 2)][3]),
                                                   output_shape=Model_Dict[model_name][str(layer_count + 2)][4])
                        prunetensor = activationSlidePrune(temp_out, r_pre_layer[model_name][layer_count + 1],
                                                           p_pre_layer[model_name][layer_count + 1], pattern='test')
                        Model_Dict[model_name][str(layer_count + Model_Dict[model_name]['thre']+1)] = prunetensor
                       
                    else:
                        min_val, max_val = temp_out.min(), temp_out.max()
                        min_val, max_val = min_val.item(), max_val.item()
                        scale_temp = (max_val - min_val) / (2 ** bits - 1)
                        temp_out = torch.round(temp_out / scale_temp)
                        temp_out = image_to_column(temp_out, (
                            Model_Dict[model_name][str(layer_count + 1)][2], Model_Dict[model_name][str(layer_count + 1)][2]),
                                                   stride=(Model_Dict[model_name][str(layer_count + 1)][3],
                                                           Model_Dict[model_name][str(layer_count + 1)][3]),
                                                   output_shape=Model_Dict[model_name][str(layer_count + 1)][4])
                        prunetensor = activationSlidePrune(temp_out, r_pre_layer[model_name][layer_count + 1],
                                                           p_pre_layer[model_name][layer_count + 1], pattern='test')
                        Model_Dict[model_name][str(layer_count + Model_Dict[model_name]['thre'])] = prunetensor
                       
                    if layer_count in resnet_Dict['need_reserve']:
                        resnet_reserve_dict[resnet_count] = o_8_t
                if layer_count in [5,10,15]:
                    BN_size = Model_Dict[model_name][str(layer_count + 1)][5]
                    bn = nn.BatchNorm2d(BN_size)
                    load_dict = np.load('my_file.npy', allow_pickle=True).item()
                    bn.bias.data = load_dict[str(model_name)][str(layer_count + 1)]['bias']
                    bn.num_batches_tracked.data = load_dict[str(model_name)][str(layer_count + 1)][
                        'num_batches_tracked']
                    bn.running_mean.data = load_dict[str(model_name)][str(layer_count + 1)]['running_mean']
                    bn.running_var.data = load_dict[str(model_name)][str(layer_count + 1)]['running_var']
                    bn.weight.data = load_dict[str(model_name)][str(layer_count + 1)]['weight']
                    o_8_t = bn(o_8_t)
                    resnet_reserve_dict[resnet_count] = o_8_t

                    temp_out = copy.deepcopy(o_4)
                    temp_out = temp_out.reshape(a)
                    temp_out = temp_out.permute(3, 0, 1, 2)
                    shape_cal = [temp_out.shape[0], temp_out.shape[1], temp_out.shape[2],
                                 temp_out.shape[3]]
                    logic_temp = torch.ones_like(temp_out)
                    need_cal = torch.logical_and(logic_temp, temp_out)  
            else:
                resnet_count = 0
            if (compute_mode == 0):
                if bias is not None:
                    o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input) + (
                        torch.unsqueeze(bias_8*scale_b, 1))
                else:
                    o_8 = torch.einsum('ij,jk->ik', weight_8 * scale_w, input)
                return o_8
        if (compute_mode == 1 or compute_mode == 2):
            for i in range(o_4.shape[0]):
                w1 = weight_8[i].view(1, len(weight_8[i]))
                if bias is not None:
                    o_8_vv = torch.einsum('ij,jk->ik', w1 * scale_w, input) + bias_8[i] * scale_b
                else:
                    o_8_vv = torch.einsum('ij,jk->ik', w1 * scale_w, input)
                w1_4 = weight_4[i].view(1, len(weight_4[i]))
                if bias is not None:
                    o_4_vv = torch.einsum('ij,jk->ik', w1_4 * scale_w, input_4 * scale_i) + bias_4[i] * scale_b
                else:
                    o_4_vv = torch.einsum('ij,jk->ik', w1_4 * scale_w, input_4 * scale_i)
                o_8_vv[:, (o_4[i] <= 0)] = 0
                o_4_vv[:, (o_4[i] > 0)] = 0
                o = o_8_vv + o_4_vv
                o_4[i] = o
            return o_4
