import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
import random
import math
import copy
import numpy as np
def determine_padding(filter_shape, output_shape="same"):
    '''
    :param filter_shape:
    :param output_shape:
    :return:
    '''
    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))
    else:
        pad_h1 = output_shape[0]
        pad_h2 = output_shape[0]
        pad_w1 = output_shape[1]
        pad_w2 = output_shape[1]

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)# Add padding to the image
    images_padded = torch.nn.functional.pad(images, [pad_w[0],pad_w[1],pad_h[0],pad_h[1]], mode='constant')# Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    # cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    cols = cols.permute(1, 2, 0).reshape(filter_height * filter_width * channels, -1)

    return cols

def get_im2col_indices(images_shape, filter_shape, padding, stride=(1,1)):  # stride:(H,W)
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride[0] + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)

class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):  # 跳过伪量化这一层的梯度计算，让梯度直接流到前一层
        grad_input = grad_output.clone()
        return grad_input

# A(特征)量化
class ActivationQuantizer(nn.Module):
    def __init__(self):
        super(ActivationQuantizer, self).__init__()

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input,bits,r_or_c = None,pattern=None,ratio=None,filter_shape=None,stride=None,padding=None,n_filters=None,batch_size=None,input_shape=None):
        self.bits = bits
        self.ratio = ratio
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.n_filters = n_filters
        self.input_shape = input_shape
        self.pattern = pattern
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            # return output*scale
            min_val, max_val = input.min(), input.max()
            min_val, max_val = min_val.item(), max_val.item()
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            output = self.round(input / scale)
            if pattern == 'Conv':
                self.X_col = image_to_column(output, self.filter_shape, stride=self.stride, output_shape=self.padding)
                if self.ratio != 0:
                    self.X_col = self.activationSlidePrune(self.X_col, self.ratio,r_or_c)
                #留下高4bit的数据
                self.X_col = self.X_col.to('cpu').numpy()
                self.X_col = np.trunc((np.trunc(self.X_col / 16)) * 16)

                self.X_col = scale * self.X_col  # 再反量化回float32
                return self.X_col
            else:
                output = output * scale
                return output
    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride[0] + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride[0] + 1
        return self.n_filters, int(output_height), int(output_width)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def compressionRateStatistics(self,input,andSum,compareRatio):
        pruneNumber = 0
        zerosNumber = 0
        for i in range(input.shape[0]):
            if andSum[i] == 0:
                zerosNumber += 1
            if andSum[i] != 0 and andSum[i] <= compareRatio:
                pruneNumber += 1
        print('pruneNumberRatio=', pruneNumber / (input.shape[0]))
        print('zerosNumberRatio=', zerosNumber / (input.shape[0]))

    def accuracyTest(self,andSum):
        for i in range(len(andSum)):
            print(i,andSum[i])

    def activationSlidePrune(self,input,ratio,r_or_c):
        matrixOne = torch.ones(input.shape,device='cuda:0')  # 设置一个全1矩阵
        # x = copy.deepcopy(input)
        x = torch.clone(torch.detach(input))
        andOp = torch.logical_and(matrixOne,x)  # 进行与操作
        if r_or_c == 1:
            andSum_row = torch.sum(andOp,dim=1)  # 每行的数据进行一个相加
            list_queue = torch.sort(andSum_row)
            num = torch.floor(torch.tensor(len(list_queue.values)*ratio[0]))
            r = list_queue.values[int(num)]
            pruneTensor_row = torch.zeros_like(andSum_row)
            pruneTensor_row[(andSum_row < r),] = 1
            aaaa = torch.sum(pruneTensor_row)
            input[(andSum_row < r),] = 0
            print("只剪行:",aaaa)
            #以上进行行剪枝
        elif r_or_c == 2:
            #列剪枝
            # print('执行列剪枝，ratio=', ratio)
            andSum_column = torch.sum(andOp, dim=0)  # 每行的数据进行一个相加
            # q = (sum(andSum_column) // len(andSum_column)) * ratio
            list_queue = torch.sort(andSum_column)
            num = torch.ceil(torch.tensor(len(list_queue.values)*ratio[1]))
            r = list_queue.values[int(num)]
            pruneTensor_row = torch.zeros_like(andSum_column)
            pruneTensor_row[(andSum_column < r),] = 1
            aaaa = torch.sum(pruneTensor_row)
            input[:,(andSum_column < r)] = 0
            print("只剪列:", aaaa // 32)
            # 以上进行列剪枝
        else:
            # print('执行行列剪枝，ratio=', ratio)
            andSum_row = torch.sum(andOp, dim=1)
            andSum_column = torch.sum(andOp, dim=0)

            list_queue_r1 = torch.sort(andSum_row)
            list_queue_r2 = torch.sort(andSum_column)

            num_r1 = torch.ceil(torch.tensor(len(list_queue_r1.values)*ratio[0]))
            num_r2 = torch.ceil(torch.tensor(len(list_queue_r2.values)*ratio[1]))

            r1 = list_queue_r1.values[int(num_r1)]
            r2 = list_queue_r2.values[int(num_r2)]

            pruneTensor_row = torch.zeros_like(andSum_row)
            pruneTensor_row[(andSum_row < r1),] = 1
            aaaa1 = torch.sum(pruneTensor_row)

            pruneTensor_column = torch.zeros_like(andSum_column)
            pruneTensor_column[(andSum_column < r2),] = 1
            aaaa2 = torch.sum(pruneTensor_column)

            input[(andSum_row < r1),] = 0
            input[:, (andSum_column < r2)] = 0
            print("剪的行:", aaaa1)
            print("剪的列:", aaaa2 // 32)


        # lens = len(zeroTensor)
        # zeroRatio = (sum(zeroTensor), float(sum(zeroTensor)) / lens)
        # pruneRatio = (sum(pruneTensor) - sum(zeroTensor), float(sum(pruneTensor) - sum(zeroTensor)) / lens)
        # input[(andSum_row<=p),] = 0

        return input

        # andSum = torch.sum(andOp,dim=1)  # 每行的数据进行一个相加
        # # self.accuracyTest(andSum)
        # p = (sum(andSum) // len(andSum))*ratio
        # # self.compressionRateStatistics(input, andSum, p)
        # zeroTensor = torch.zeros_like(andSum)
        # pruneTensor = torch.zeros_like(andSum)
        # pruneTensor[(andSum <= p),] = 1
        # zeroTensor[(andSum == 0),] = 1
        # lens = len(zeroTensor)
        # zeroRatio = (sum(zeroTensor), float(sum(zeroTensor)) / lens)
        # pruneRatio = (sum(pruneTensor) - sum(zeroTensor), float(sum(pruneTensor) - sum(zeroTensor)) / lens)
        # input[(andSum<=p),] = 0
        #
        # return input

# W(权重)量化
class WeightQuantizer(nn.Module):
    def __init__(self):
        super(WeightQuantizer, self).__init__()

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input,bits,pattern,n_filters):
        self.bits = bits
        self.n_filters = n_filters
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            # min_val,max_val = input.min(),input.max()
            # min_val,max_val = min_val.item(),max_val.item()
            # scale = (max_val-min_val)/(2**self.bits - 1)
            # output = self.round(input/scale)
            # output = output.reshape((self.n_filters, -1))
            # output = scale*output
            # 权重img2col展开
            '''
            if pattern == 'Conv':
                output = output.reshape((self.n_filters, -1))
                output = output.to('cpu').numpy()
                output = np.trunc((np.trunc(output / 16)) * 16)
            output = scale*output   # 再反量化回32位
            '''
            # output = input
            min_val, max_val = input.min(), input.max()
            min_val, max_val = min_val.item(), max_val.item()
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            output = self.round(input / scale)
            if pattern == 'Conv':
                output = output.reshape((self.n_filters, -1))
                # 留下高4bit数据
                output = output.to('cpu').numpy()
                output = np.trunc((np.trunc(output / 16)) * 16)

            output = scale * output  # 先全部量化成8位
            # output = input
        return output
        # 返回的output是img2col展开后的权重矩阵