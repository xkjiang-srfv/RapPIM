import numpy as np
import torch
import copy
from itertools import chain


def oneCal_original(input):
    # a1 = torch.sum(input)
    input = input.to('cpu')
    input = input.detach().numpy()
    input = input.tolist()
    # a2 = sum(map(sum,input))
    input = list(chain.from_iterable(input))
    input = [*map(int, input)]
    # a3 = sum(input)
    # print("a3==",a3)
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    print("Quantize:SRE操作时，本层一共有{}个1".format(s1))

def oneCal(input):
    # a1 = torch.sum(input)
    input = input.to('cpu')
    input = input.detach().numpy()
    input = input.tolist()
    # a2 = sum(map(sum,input))
    input = list(chain.from_iterable(input))
    input = [*map(int, input)]
    # a3 = sum(input)
    # print("a3==",a3)
    input = [*map(bin, input)]
    q = "".join(input)
    s1 = q.count("1")
    return s1
    # print("Quantize:SRE操作时，本层一共有{}个1".format(s1))

