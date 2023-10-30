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

