import torch
import math
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
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))
    else:
        pad_h1 = output_shape[0]
        pad_h2 = output_shape[0]
        pad_w1 = output_shape[1]
        pad_w2 = output_shape[1]

        return (pad_h1, pad_h2), (pad_w1, pad_w2)
def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)  # Add padding to the image
    images_padded = torch.nn.functional.pad(images, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]],
                                            mode='constant')  # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    # cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    cols = cols.permute(1, 2, 0).reshape(filter_height * filter_width * channels, -1)

    return cols
def get_im2col_indices(images_shape, filter_shape, padding, stride=(1, 1)):  # stride:(H,W)
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

def activationSlidePrune(input, ratio, r_or_c, pattern='Train'):
    matrixOne = torch.ones(input.shape, device='cuda:0')  # 设置一个全1矩阵
    # x = copy.deepcopy(input)
    x = torch.clone(torch.detach(input))
    andOp = torch.logical_and(matrixOne, x)  # 进行与操作

    if r_or_c == 1:
        andSum_row = torch.sum(andOp, dim=1)  # 每行的数据进行一个相加
        list_queue = torch.sort(andSum_row)
        num = torch.floor(torch.tensor(len(list_queue.values) * ratio[0]))
        r = list_queue.values[int(num)]
        pruneTensor_row = torch.zeros_like(andSum_row)
        if r == 0:
            pruneTensor_row[(andSum_row <= r),] = 1
        else:
            pruneTensor_row[(andSum_row < r),] = 1
        # aaaa = torch.sum(pruneTensor_row)
        # print("只剪行:", aaaa)
        # print("剪枝比例为：", aaaa / len(pruneTensor_row))
        if pattern == 'test':
            return (1,pruneTensor_row)
        else:
            input[(andSum_row < r),] = 0
            return input

    elif r_or_c == 2:
        # 列剪枝
        # print('执行列剪枝，ratio=', ratio)
        andSum_column = torch.sum(andOp, dim=0)  # 每行的数据进行一个相加
        # q = (sum(andSum_column) // len(andSum_column)) * ratio
        list_queue = torch.sort(andSum_column)
        num = torch.floor(torch.tensor(len(list_queue.values) * ratio[1]))
        r = list_queue.values[int(num)]
        pruneTensor_column = torch.zeros_like(andSum_column)
        if r == 0:
            pruneTensor_column[(andSum_column <= r),] = 1
        else:
            pruneTensor_column[(andSum_column < r),] = 1
        # aaaa = torch.sum(pruneTensor_column)
        # print("只剪列:", aaaa // 64)
        # print("剪枝比例为：", aaaa / len(pruneTensor_column))
        if pattern == 'test':
            return (2,pruneTensor_column)
        else:
            input[:, (andSum_column < r)] = 0
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
        # aaaa1 = torch.sum(pruneTensor_row)

        pruneTensor_column = torch.zeros_like(andSum_column)
        if r2 == 0:
            pruneTensor_column[(andSum_column <= r2),] = 1
        else:
            pruneTensor_column[(andSum_column < r2),] = 1

        # aaaa2 = torch.sum(pruneTensor_column)
        # print("剪的行:", aaaa1)
        # print("剪的列:", aaaa2 // 64)
        # print("行剪枝比例为：", aaaa1 / len(pruneTensor_row))
        # print("列剪枝比例为：", aaaa2 / len(pruneTensor_column))
        if pattern == 'test':
            return (3,pruneTensor_row, pruneTensor_column)
        else:
            input[(andSum_row < r1),] = 0
            input[:, (andSum_column < r2)] = 0
            return input
