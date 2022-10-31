from ActivationPrune import *
from WeightPrune import weightPruneModelOp
import os
from Op import Op
import torch
import os
import collections
torch.set_printoptions(threshold=5e3,edgeitems=15)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = 'VGG8'  # 确定模型名称
    batch_size = 2  # 确定批训练图片数目
    img_size = 32  # 确定单张图片大小
    ratio = 1  # 确定输入特征图剪枝比率
    epochA = 1  # 确定针对输入特征图剪枝重训练轮数或原始模型（不掺杂任何剪枝训练）轮数
    epochAW = 10  # 确定针对卷积核聚类剪枝重训练轮数
    quantize_w = 8
    quantize_i = 8
    weightParameter = (4/2)
    LinearParameter = (4/3)
    r_pre_layer = {
        'LeNet':[(0.2,0.8),(0.5,0.2),(0.5,0.5)],
        'AlexNet':[(0.6, 0.9), (0.65, 0.4), (0.5, 0.0), (0.55, 0.0), (0.5, 0.0)],
        'VGG16':[(0.2,0.3),(0.4,0.4),(0.3,0.4),(0.6,0.4),(0.4,0.4),(0.6,0.4),(0.6,0.5),(0.4,0.5),(0.6,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4)],
        'VGG8':[(0.2,0.3),(0.6,0.2),(0.55,0.4),(0.6,0.4),(0.55,0.4),(0.6,0.4),(0.55,0.4)],
        'ResNet':[(0.4,0.4) for i in range(20)],
        'NewResNet':[(0.5,0.5) for i in range(20)],
        'ZFNet':[(0.5,0.5) for i in range(5)]
    }
    p_pre_layer = {
        'LeNet':[2,3,1],
        'AlexNet':[2,3,1,1,1],
        'VGG16':[2,1,1,1,1,1,1,1,1,1,1,1,1],
        'VGG8':[2,3,1,1,1,1,1],
        'ResNet':[1 for i in range(20)],
        'NewResNet':[1 for i in range(20)],
        'ZFNet':[1 for i in range(5)]
    }
    # 1: 行剪枝
    # 2: 列剪枝
    # 3: 行列剪枝
    '''
    一共设置有七种针对模型的操作
    1. operation = 'trainInitialModel'，意为训练初始模型，此时不参杂任何剪枝操作，单纯训练初始模型
    2. operation = 'onlyActivationPruneWithRetrain'，意为只针对输入特征图进行剪枝，并进行重训练
    3. operation = 'onlyWeightPruneWithRetrain'，意为只针对权重值进行聚类剪枝，并进行重训练
    4. operation = 'activationRetrainAfterWeightRetrain'，意为此时我已经单独完成了输入特征图剪枝的行为，保存了模型，此时我想再进行权重聚类剪枝
    5. operation = 'activationWeightPruneWithRetrain'，意为对输入特征图剪枝并进行重训练，对其生成的模型权重进行聚类剪枝并进行重训练
    6. operation = 'onlyActivationPruneTest'，意为只针对输入特征图剪枝后的模型进行inferernce，测试模型精度
    7. operation = 'activationWeightPruneTest'，意为针对输入特征图与权重聚类剪枝后的模型进行inference，测试模型精度
    8. operation = 'unstructureWeightPrune'
    9. operation = 'trainQuantizeModel',训练完初始模式后进行量化操作
    '''


    operation = 'trainInitialModel'
    lr_rate = 0.0005
    Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter,quantize_w,quantize_i,lr_rate,r_pre_layer,p_pre_layer)
    '''
    目录说明
    -pth
        --modelName
            ---ratio=0
                ----Activation:存放不经过任何剪枝的初始模型
                ----Weight:存放只经过权重聚类剪枝后的初始模型
            ---ratio=0.1
                ----Activation:存放经过输入特征图剪枝后的模型
                ----ActivationWeight:存放经过输入特征图剪枝后又进行权重聚类剪枝后的模型
    '''
