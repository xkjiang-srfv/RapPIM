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
    print("第",layerCount,"层的参数个数=", x.shape[0]*x.shape[1],"当前层的0值个数等于=",zeroNumber)

def oneCal_original(input):
    # a1 = torch.sum(input)
    input = input.to('cpu')
    dim1 = input.shape[0]
    dim2 = input.shape[1]
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))   # 将二维数组展开成一维，[[1,1,1],[2,2,2],[3,3,3]]->[1,1,1,2,2,2,3,3,3]
    input = [*map(int, input)]
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    print("行列剪枝前（或SRE），本层一共有{}个1".format(s1))

    xk_list = []
    for i in range(0, dim1 * dim2, dim2):
        q = "".join(input[i:i+dim2])
        s = q.count("1")
        xk_list.append(s)
    print("行列剪枝前（或SRE），本层一共有{}个1".format(sum(xk_list)))
    print("本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
    print("本层中拥有最少1的行中1的个数为{}".format(min(xk_list)))
    print("本层中平均每行1的个数为{}".format(sum(xk_list)/len(xk_list)))
    print("********************************************")

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
    # print("当前mode为：",mode)
    if(mode == 'OnlyRCP'):
        # print("经行列剪枝后，本层一共有{}个1".format(sum(xk_list)))
        # print("经行列剪枝后，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("本层中非零行平均每行1的个数为{}".format(sum(xk_list) / no_zero_count))
        # print(sum(xk_list))
        # print(max(xk_list))
        print(sum(xk_list) / no_zero_count)
        # print("********************************************")
    elif(mode=="RCP_Skip"):
        # print("经行列剪枝后，只考虑高位数据，本层一共有{}个1".format(sum(xk_list)))
        # print("经行列剪枝后，只考虑高位数据，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("经行列剪枝后，只考虑高位数据，本层中非零行平均每行1的个数为{}".format(sum(xk_list) / no_zero_count))
        # print(sum(xk_list))
        # print(max(xk_list))
        print(sum(xk_list) / no_zero_count)
        # print("********************************************")
    else:
        # print("SRE模式下，本层一共有{}个1".format(sum(xk_list)))
        # print("SRE模式下，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("SRE模式下，本层中平均每行1的个数为{}".format(sum(xk_list) / len(xk_list)))
        # print(sum(xk_list))
        # print(max(xk_list))
        print(sum(xk_list) / len(xk_list))
        # print("********************************************")
    return s1

# 全链接层是64*512，要把他转换成512*64，64才是真正输入到Crossbar中的列数
# FC层未执行任何行列剪枝，因此除上的都是len(xk_list)
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
    # print("当前mode为：",mode)
    if (mode == 'OnlyRCP'):
        # print("经行列剪枝后，本层一共有{}个1".format(sum(xk_list)))
        # print("经行列剪枝后，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("本层中非零行平均每行1的个数为{}".format(sum(xk_list) / no_zero_count))
        print(sum(xk_list))
        # print(max(xk_list))
        # print(sum(xk_list) / len(xk_list))
        # print("********************************************")
    elif (mode == "RCP_Skip"):
        # print("经行列剪枝后，只考虑高位数据，本层一共有{}个1".format(sum(xk_list)))
        # print("经行列剪枝后，只考虑高位数据，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("经行列剪枝后，只考虑高位数据，本层中非零行平均每行1的个数为{}".format(sum(xk_list) / no_zero_count))
        # print(sum(xk_list))
        # print(max(xk_list))
        print(sum(xk_list) / len(xk_list))
        # print("********************************************")
    else:
        # print("SRE模式下，本层一共有{}个1".format(sum(xk_list)))
        # print("SRE模式下，本层中拥有最多1的行中1的个数为{}".format(max(xk_list)))
        # print("SRE模式下，本层中平均每行1的个数为{}".format(sum(xk_list) / len(xk_list)))
        # print(sum(xk_list))
        # print(max(xk_list))
        print(sum(xk_list) / len(xk_list))
        # print("********************************************")
    return s1
