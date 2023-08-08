from ActivationPrune import highBitPruneModel

def AccuTest(model_name,ratio,img_size,mode,num,r_pre_layer,p_pre_layer):
    for i in range(num):
        paraDict = {
            'model_name': model_name,
            'ratio': ratio,
            'batch_size': 64,
            'img_size': img_size,
            'pattern': 'test',
            'epoch': 100,
            'quantize': True,
            'quantize_w': 8,
            'quantize_i': 8,
            'mode': mode,
            'cycle': i,
            # 0:only High Bit Forecast
            # 1:only High Bit compute
            # 2:both of all
        }
        # dataloaders, dataset_sizes = getDataSet(paraDict['model_name'], paraDict['batch_size'], paraDict['img_size'])
        highBitPruneModel(paraDict['model_name'],paraDict['batch_size'],paraDict['img_size'],paraDict['pattern'],paraDict['ratio'],
                               paraDict['epoch'],paraDict['quantize'],paraDict['quantize_w'],paraDict['quantize_i'],
                          paraDict['mode'],paraDict['cycle'],r_pre_layer,p_pre_layer)
if __name__ == '__main__':
    model_name = 'NewResNet'
    ratio = 1
    img_size = 32
    mode = 2
    num = 1
    r_pre_layer = {
        'LeNet': [(0.2, 0.8), (0.5, 0.2), (0.5, 0.5)],
        'AlexNet': [(0.6, 0.9), (0.65, 0.4), (0.5, 0.0), (0.55, 0.0), (0.5, 0.0)],
        'VGG16': [(0.2,0.3),(0.4,0.4),(0.3,0.4),(0.6,0.4),(0.4,0.4),(0.6,0.4),(0.6,0.5),(0.4,0.5),(0.6,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4),(0.85,0.4)],
        'VGG8': [(0.2, 0.3), (0.6, 0.2), (0.55, 0.4), (0.6, 0.4), (0.55, 0.4), (0.6, 0.4), (0.55, 0.4)],
        'ResNet': [(0.4, 0.4) for i in range(20)],
        'NewResNet': [(0.5,0.5) for i in range(20)],
        'ZFNet':[(0.35,0.35),(0.5,0.3),(0.5,0),(0.5,0),(0.5,0)]
    }
    # r_pre_layer = {
    #     'LeNet': [(0,0) for i in range(3)],
    #     'AlexNet': [(0,0) for i in range(5)],
    #     'VGG16': [(0,0) for i in range(13)],
    #     'VGG8': [(0,0) for i in range(7)],
    #     'ResNet': [(0,0) for i in range(20)],
    #     'NewResNet': [(0,0) for i in range(20)],
    #     'ZFNet': [(0,0) for i in range(5)]
    # }
    p_pre_layer = {
        'LeNet': [2, 3, 1],
        'AlexNet': [2, 3, 1, 1, 1],
        'VGG16': [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'VGG8': [2, 3, 1, 1, 1, 1, 1],
        'ResNet': [1 for i in range(20)],
        'NewResNet' : [1 for i in range(20)],
        'ZFNet':[2,1,1,1,1]
    }
    AccuTest(model_name,ratio,img_size,mode,num,r_pre_layer,p_pre_layer)
