from ActivationPrune import activationPruneModelOp
from WeightPrune import weightPruneModelOp
import os
def makeDir(model_name,ratio):  
    if not os.path.exists('./pth/' + model_name + '/ratio=' + str(ratio)):
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Activation')
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Quantize')
        if ratio == 0:  
            os.makedirs('./pth/' + model_name + '/ratio=0/' + 'Weight')
            os.makedirs('./pth/' + model_name + '/ratio=0/' + 'DynamicPrune')
        else:
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')

def Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer):
    if operation == 'trainInitialModel': 
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
        makeDir(model_name,ratio)  
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'trainQuantizeModel': 
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
        makeDir(model_name,ratio) 
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'onlyActivationPruneWithRetrain': 
        patternA = 'retrain'
        quantize = True
        makeDir(model_name,ratio)
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA,quantize,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)

    if operation == 'onlyActivationPruneTest':
        quantize = False
        patternA = 'test'
        makeDir(model_name, ratio)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA, quantize, quantize_w,quantize_i, lr_rate, r_pre_layer, p_pre_layer)

    if operation == 'onlyWeightPruneWithRetrain':   
        patternW = 'train'  
        quantize = False
        ratio = 0
        makeDir(model_name,ratio)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter,LinearParameter,quantize,quantize_w,quantize_i)

    if operation == 'activationWeightPruneWithRetrain':
        quantize = False
        patternA = 'retrain'
        patternW = 'retrain' 
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

    if operation == 'unstructureWeightPrune':  
        quantize = False
        patternW = 'DPTrain'
        ratio = 0
        makeDir(model_name,ratio)  
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter,quantize,quantize_w,quantize_i)