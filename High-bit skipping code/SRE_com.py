import numpy as np
import torch
import copy
from itertools import chain

def countSparsity(input,layerCount):
    matrixOne = torch.ones(input.shape, device='cuda:0')
    x = torch.clone(torch.detach(input))
    andOp = torch.logical_and(matrixOne, x)
    zeroNumber = x.shape[0]*x.shape[1] - torch.sum(andOp)
    nowLayerSparsty = zeroNumber/torch.sum(andOp)

def oneCal_original(input):
    # a1 = torch.sum(input)
    input = input.to('cpu')
    dim1 = input.shape[0]
    dim2 = input.shape[1]
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))   
    input = [*map(int, input)]
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")

    xk_list = []
    for i in range(0, dim1 * dim2, dim2):
        q = "".join(input[i:i+dim2])
        s = q.count("1")
        xk_list.append(s)
    

def oneCal(input,mode='RCP_Skip'):
    # a1 = torch.sum(input)
    input = input.to('cpu')
    dim1 = input.shape[0]
    dim2 = input.shape[1]
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))
    input = [*map(int, input)]
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    xk_list = []
    for i in range(0, dim1 * dim2, dim2):
        q = "".join(input[i:i+dim2])
        s = q.count("1")
        xk_list.append(s)
    no_zeros_value = [num for num in xk_list if num!=0]
    no_zero_count = len(no_zeros_value)
    if(mode == 'OnlyRCP'):
        print(sum(xk_list) / no_zero_count)
    elif(mode=="RCP_Skip"):
        print(sum(xk_list) / no_zero_count)
    else:
        print(sum(xk_list) / len(xk_list))
    return s1


def oneCal_FC(input_origin,mode='RCP_Skip'):
    # a1 = torch.sum(input)
    input = copy.deepcopy(input_origin)
    input = torch.transpose(input,dim0=0,dim1=1)
    input = input.to('cpu')
    dim1 = input.shape[0]
    dim2 = input.shape[1]
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))
    input = [*map(int, input)]
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    xk_list = []
    for i in range(0, dim1 * dim2, dim2):
        q = "".join(input[i:i + dim2])
        s = q.count("1")
        xk_list.append(s)
    no_zeros_value = [num for num in xk_list if num != 0]
    no_zero_count = len(no_zeros_value)
    if (mode == 'OnlyRCP'):
        print(sum(xk_list))
    elif (mode == "RCP_Skip"):
        print(sum(xk_list) / len(xk_list))
    else:
        print(sum(xk_list) / len(xk_list))
    return s1
