import numpy as np
import torch
import copy
from itertools import chain
def operate(x):
    # x = bin(int(x))
    # x = x[2:len(x)]
    x += 1
    return x


def oneCalRow(input,pruneTensor=None):
    ones = torch.zeros(len(input))
    input = input.to('cpu')
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))
    # for i in range(len(input)):
    #     input[i] = [*map(int, input[i])]
    #     input[i] = [*map(bin,input[i])]
    #     q = "".join(input[i])
    #     ones[i] = q.count("1")
    input = [*map(int,input)]
    input = [*map(bin,input)]
    q = "".join(input)
    s1 = q.count("1")
    # s1 = float(sum(ones))
    # ones[(pruneTensor==0),] = 0
    # s2 = float(sum(ones))  # 剪掉的
    # s3 = s1 - s2  #剩余的
    print("SRE（行）操作时，本层一共有{}个1，占全部位数的{}".format(s1,s1/(len(input)*len(input[0])*8)))
    # print("剪行时，该行一共有{}个1，剪掉了{}个,剩余{}个,剪掉的比例为{}".format(s1,s2,s3,(s2/s1)))

def oneCalColumn(input,pruneTensor):
    input = input.T
    ones = torch.zeros(len(input))
    input = input.to('cpu')
    input = input.detach().numpy()
    input = input.tolist()
    input = list(chain.from_iterable(input))
    # for i in range(len(input)):
    #     input[i] = [*map(int, input[i])]
    #     input[i] = [*map(bin, input[i])]
    #     q = "".join(input[i])
    #     ones[i] = q.count("1")
    input = [*map(int, input)]
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    # ones[(pruneTensor == 0),] = 0
    # s2 = float(sum(ones))  # 剪掉的
    # s3 = s1 - s2  # 剩余的
    print("SRE（列）时，本层一共有{}个1，占全部位数的{}".format(s1,s1/(len(input)*len(input[0])*8)))
    # print("剪列时，该列一共有{}个1，剪掉了{}个,剩余{}个,剪掉的比例为{}".format(s1, s2, s3, (s2 / s1)))

def oneCal(input):
    a1 = torch.sum(input)
    input = input.to('cpu')
    input = input.detach().numpy()
    input = input.tolist()
    a2 = sum(map(sum,input))
    input = list(chain.from_iterable(input))
    input = [*map(int, input)]
    a3 = sum(input)
    print("a3==",a3)
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    print("SRE操作时，本层一共有{}个1".format(s1))

