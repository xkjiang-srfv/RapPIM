from ActivationPrune import activationPruneModelOp
from WeightPrune import weightPruneModelOp
import os
def makeDir(model_name,ratio):  # 创建文件夹
    if not os.path.exists('./pth/' + model_name + '/ratio=' + str(ratio)):
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Activation')
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Quantize')
        if ratio == 0:  # ratio=0只有两种情况，一是训练初始模型的时候,二是训练量化模型
            os.makedirs('./pth/' + model_name + '/ratio=0/' + 'Weight')
            os.makedirs('./pth/' + model_name + '/ratio=0/' + 'DynamicPrune')
        else:
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')

def Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer):
    if operation == 'trainInitialModel':  # 训练初始模型
        patternA = 'train'
        quantize = False
        ratio = 0
        r_pre_layer = {
            'LeNet': [0 for i in range(3)],
            'AlexNet': [0 for i in range(5)],
            'VGG16': [0 for i in range(13)],
            'VGG8': [0 for i in range(7)],
            'ResNet': [0 for i in range(20)],
            'NewResNet':[0 for i in range(20)],
            'ZFNet': [0 for i in range(5)]
        }
        makeDir(model_name,ratio)  # 判断是否为训练初始模型并创建相应的文件夹
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'trainQuantizeModel':  # 训练初始模型
        patternA = 'retrain'
        ratio = 0
        quantize = True
        r_pre_layer = {
            'LeNet': [0 for i in range(3)],
            'AlexNet': [0 for i in range(5)],
            'VGG16': [0 for i in range(13)],
            'VGG8': [0 for i in range(7)],
            'ResNet': [0 for i in range(19)],
            'NewResNet': [0 for i in range(20)],
            'ZFNet': [0 for i in range(5)]
        }
        makeDir(model_name,ratio)  # 判断是否为训练初始模型并创建相应的文件夹
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'onlyActivationPruneWithRetrain':  # 只进行输入特征图的剪枝，不进行权重的聚类剪枝
        patternA = 'retrain'
        quantize = True
        makeDir(model_name,ratio)
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'onlyActivationPruneTest':
        quantize = False
        patternA = 'test'
        makeDir(model_name, ratio)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA, quantize, quantize_w,quantize_i, lr_rate, r_pre_layer, p_pre_layer)

    if operation == 'onlyWeightPruneWithRetrain':   # 这有bug
        patternW = 'train'  # patternW='retrain'是读入初始模型进行权重聚类剪枝压缩再重训练
        quantize = False
        ratio = 0
        makeDir(model_name,ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter,LinearParameter,quantize,quantize_w,quantize_i)

    if operation == 'activationWeightPruneWithRetrain':
        quantize = False
        patternA = 'retrain'
        patternW = 'retrain'  # patternW='retrain'是读入已经经过输入特征图剪枝的模型进行压缩在重训练
        makeDir(model_name, ratio)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter,quantize,quantize_w,quantize_i)

    if operation == 'activationWeightPruneTest':
        quantize = False
        patternW = 'test'
        makeDir(model_name, ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter,quantize,quantize_w,quantize_i)

    if operation == 'activationRetrainAfterWeightRetrain':
        quantize = False
        patternW = 'retrain'
        makeDir(model_name, ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter,quantize,quantize_w,quantize_i,operation='activationRetrainAfterWeightRetrain')

    if operation == 'unstructureWeightPrune':  # 训练初始模型
        quantize = False
        patternW = 'DPTrain'
        ratio = 0
        makeDir(model_name,ratio)  # 判断是否为训练初始模型并创建相应的文件夹
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter,quantize,quantize_w,quantize_i)