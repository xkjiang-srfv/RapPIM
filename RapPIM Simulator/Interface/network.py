# -*-coding:utf-8-*-
import copy
import math
import collections
import numpy as np
import configparser


class NetworkGraph:
    def __init__(self, hardware_config, layer_config_list, quantize_config_list, input_index_list, reuse_config_list, prune_config_list, input_params,activation_config_list):
        super(NetworkGraph, self).__init__()
        # same length for layer_config_list , quantize_config_list and input_index_list
        assert len(layer_config_list) == len(quantize_config_list)
        assert len(layer_config_list) == len(input_index_list)

        self.hardware_config = copy.deepcopy(hardware_config)
        self.layer_config_list = copy.deepcopy(layer_config_list)
        self.quantize_config_list = copy.deepcopy(quantize_config_list)
        self.reuse_config_list = copy.deepcopy(reuse_config_list)
        self.prune_config_list = copy.deepcopy(prune_config_list)
        self.input_index_list = copy.deepcopy(input_index_list)
        self.input_params = copy.deepcopy(input_params)
        self.activation_config_list = copy.deepcopy(activation_config_list)
        xbar_config = configparser.ConfigParser() 
        xbar_config.read('../SimConfig.ini', encoding='UTF-8')
        self.OU_size = list(map(int, xbar_config.get('Crossbar level', 'OU_Size').split(',')))
        self.net_info = []
        self.net_bit_weights = []

    def set_reuse_and_prune_ratio(self):
        layer_number = 0
        for i in range(0, len(self.layer_config_list)):
            if self.net_info[i]['type'] == 'conv' or self.net_info[i]['type'] == 'fc':
                self.net_info[i]['reuse_ratio'] = 1.0 - self.reuse_config_list[layer_number]
                self.net_info[i]['prune_ratio'] = 1.0 - self.prune_config_list[layer_number]
                self.net_info[i]['activation_ratio'] =  self.activation_config_list[layer_number]
                layer_number = layer_number + 1
                print(str(layer_number) + ' reuse_ratio: ' + str(self.net_info[i]['reuse_ratio']) + ' prune_ratio: ' + str(self.net_info[i]['prune_ratio']))

    def get_weights(self):
        net_bit_weights = []
        for i in range(0, len(self.layer_config_list)):
            bit_weights = collections.OrderedDict()
            if self.net_info[i]['type'] == 'conv' or self.net_info[i]['type'] == 'fc':
                complete_bar_row = math.ceil(self.net_info[i]['OU_row_number'] / math.floor(self.hardware_config['xbar_size'] / self.OU_size[0]))
                complete_bar_column = math.ceil(self.net_info[i]['OU_column_number'] / math.floor(self.hardware_config['xbar_size'] / self.OU_size[1]))
                self.net_info[i]['Crossbar_number'] = complete_bar_row * complete_bar_column
                for j in range(0, self.net_info[i]['Crossbar_number']):
                    bit_weights[f'split{i}_weight{j}'] = np.ones((self.hardware_config['xbar_size'], self.hardware_config['xbar_size']))  
            net_bit_weights.append(bit_weights)

        self.net_bit_weights = net_bit_weights

    def get_structure(self):
        net_info = []
        input_size = [0] * len(self.layer_config_list)
        output_size = [0] * len(self.layer_config_list)
        for i in range(0, len(self.layer_config_list)):
            layer_info = collections.OrderedDict()
            layer_info['Multiple'] = 1
            layer_info['reuse_ratio'] = 0.0
            layer_info['prune_ratio'] = 0.0
            layer_info['OU_row_number'] = 0
            layer_info['OU_column_number'] = 0
            layer_info['Crossbar_number'] = 0
            layer_info['PE_number'] = 0

            if self.layer_config_list[i]['type'] == 'conv':
                layer_info['type'] = 'conv'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_channels']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_channels']
                layer_info['Kernelsize'] = self.layer_config_list[i]['kernel_size']
                layer_info['Stride'] = self.layer_config_list[i]['stride']
                layer_info['Padding'] = self.layer_config_list[i]['padding']

                layer_info['OU_row_number'] = math.ceil(self.layer_config_list[i]['in_channels'] * self.layer_config_list[i]['kernel_size'] * self.layer_config_list[i]['kernel_size'] / self.OU_size[0])
                layer_info['OU_column_number'] = math.ceil(self.layer_config_list[i]['out_channels'] * self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'] / self.OU_size[1])

                layer_info['reuse_ratio'] = self.layer_config_list[i]['reuse_ratio']
                layer_info['prune_ratio'] = self.layer_config_list[i]['prune_ratio']
                # layer_info['activation_ratio'] = self.layer_config_list[i]['a']

                if i == 0:
                    input_size[i] = self.input_params['input_shape'][2]
                else:
                    input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = int((input_size[i] + 2 * self.layer_config_list[i]['padding'] - (self.layer_config_list[i]['kernel_size'] - 1)) / self.layer_config_list[i]['stride'])
                layer_info['Inputsize'] = [input_size[i], input_size[i]]
                layer_info['Outputsize'] = [output_size[i], output_size[i]]
                layer_info['Multiple'] = 1

            elif self.layer_config_list[i]['type'] == 'fc':
                layer_info['type'] = 'fc'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_features']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_features']

                layer_info['OU_row_number'] = math.ceil(self.layer_config_list[i]['in_features'] / self.OU_size[0])
                layer_info['OU_column_number'] = math.ceil(self.layer_config_list[i]['out_features'] * self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'] / self.OU_size[1])

                layer_info['reuse_ratio'] = self.layer_config_list[i]['reuse_ratio']
                layer_info['prune_ratio'] = self.layer_config_list[i]['prune_ratio']

            elif self.layer_config_list[i]['type'] == 'pooling':
                layer_info['type'] = 'pooling'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_channels']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_channels']
                layer_info['Kernelsize'] = self.layer_config_list[i]['kernel_size']
                layer_info['Stride'] = self.layer_config_list[i]['stride']
                layer_info['Padding'] = self.layer_config_list[i]['padding']

                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = int(input_size[i] / self.layer_config_list[i]['stride'])
                layer_info['Inputsize'] = [input_size[i], input_size[i]]
                layer_info['Outputsize'] = [output_size[i], output_size[i]]
                # layer_info['Multiple'] = math.ceil(output_size[i] / self.base_size)
                layer_info['Multiple'] = 1  

            elif self.layer_config_list[i]['type'] == 'relu':
                layer_info['type'] = 'relu'
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            elif self.layer_config_list[i]['type'] == 'view':
                layer_info['type'] = 'view'

            elif self.layer_config_list[i]['type'] == 'bn':
                layer_info['type'] = 'bn'
                layer_info['features'] = self.layer_config_list[i]['features']
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            elif self.layer_config_list[i]['type'] == 'element_sum':
                layer_info['type'] = 'element_sum'
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            else:
                assert 0, f'not support {self.layer_config_list[i]["type"]}'

            layer_info['Inputbit'] = int(self.quantize_config_list[i]['activation_bit'])
            layer_info['Weightbit'] = int(self.quantize_config_list[i]['weight_bit'])
            layer_info['Outputbit'] = layer_info['Inputbit']  
            # if i != len(self.layer_config_list) - 1:
            #     layer_info['Outputbit'] = int(self.quantize_config_list[i+1]['activation_bit'])
            # else:
            #     layer_info['Outputbit'] = int(self.quantize_config_list[i]['activation_bit'])
            layer_info['weight_cycle'] = math.ceil(self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'])
            if 'input_index' in self.layer_config_list[i]:
                layer_info['Inputindex'] = self.layer_config_list[i]['input_index']
            else:
                layer_info['Inputindex'] = [-1]
            layer_info['Outputindex'] = [1]

            net_info.append(layer_info)

        self.net_info = net_info


def get_net(hardware_config, cate, num_classes, mode,condition):
    # initial config
    if hardware_config is None:
        hardware_config = {'xbar_size': 256, 'input_bit': 8, 'weight_bit': 8, 'quantize_bit': 8}
    layer_config_list = []
    quantize_config_list = []
    input_index_list = []
    reuse_config_list = []
    prune_config_list = []
    activation_config_list = []
    assert condition in ['energy','latency']
    # layer by layer
    assert cate in ['VGG16','AlexNet','ZFNet','VGG8','NewResNet']
    if cate.startswith('AlexNet'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 96, 'kernel_size': 11, 'stride': 4, 'padding': 2,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 96, 'out_channels': 96, 'kernel_size': 3, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 96, 'out_channels': 256, 'kernel_size': 5, 'stride': 1, 'padding': 2,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 384, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 384, 'out_channels': 384, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 256 * 6 * 6, 'out_features': 4096, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': 4096, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': 10, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   
        activation_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

        if mode == 'shapePipe':
            prune_config_list = [1.0, 0.35, 0.5, 0.45, 0.5, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.241373166, 0.115736234, 0.247148793, 0.055777163, 0.035924314, 0.3359375,0.4296875,0.32421875]
            elif condition == 'energy':
                activation_config_list = [0.235960689, 0.060273944, 0.076505398, 0.019908531, 0.011179082, 0.211755117,0.184491634,0.144688129]

        if mode == 'shape' :  #
            prune_config_list = [1.0, 0.35, 0.5, 0.45, 0.5, 1.0, 1.0,1.0]
            if condition == 'latency':
                activation_config_list = [ 0.241373166,0.115736234,0.247148793,0.055777163,0.035924314,0.3359375,0.4296875,0.32421875]

            elif condition == 'energy':
                activation_config_list = [0.235960689, 0.060273944, 0.076505398, 0.019908531, 0.011179082, 0.211755117,0.184491634,0.144688129]

        if mode == 'onlyRCP':
            prune_config_list = [1.0, 0.35, 0.5, 0.45, 0.5, 1.0, 1.0,1.0]
            if condition == 'latency':
                activation_config_list = [0.454659091,0.216898684,0.386915218,0.143837833,0.107063609, 0.33203125,0.41796875,0.283203125]
            elif condition == 'energy':
                activation_config_list = [0.449669043,0.133944031,0.156383196,0.066301643,0.035558075, 0.209265391,0.181919098,0.142277241]

        if mode == 'SRE':  
            prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.49172327,0.3320634,0.305080436,0.127877681,0.063285873,0.337890625,0.419921875,0.294921875]
            elif condition == 'energy':
                activation_config_list = [0.482104782,0.109432148,0.131166422,0.051183265,0.02454356, 0.204868105,0.187413216,0.147133827]

    elif cate.startswith('ZFNet'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 48, 'kernel_size': 7, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 48, 'out_channels': 48, 'kernel_size': 3, 'stride': 2, 'padding': 1})

        layer_config_list.append({'type': 'conv', 'in_channels': 48, 'out_channels': 128, 'kernel_size': 5, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 192, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 192, 'out_channels': 192, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 192, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 128*6*6, 'out_features': 2048, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': 2048, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': 10, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        activation_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  

        if mode == 'shapePipe':
            prune_config_list = [1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.307269203, 0.256589185, 0.293438637, 0.083706376, 0.040801775, 0.318359375,0.478515625,0.45703125]
            elif condition == 'energy':
                activation_config_list = [0.302503632,0.167890152,0.140282515,0.035724179,0.011379242,0.190383487,0.315666199,0.181910515]

        if mode == 'shape':  #
            prune_config_list = [1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.307269203,0.256589185,0.293438637,0.083706376,0.040801775,0.318359375,0.478515625,0.45703125]
            elif condition == 'energy':
                activation_config_list = [0.302503632,0.167890152,0.140282515,0.035724179,0.011379242,0.190383487,0.315666199,0.181910515]

        if mode == 'onlyRCP':
            prune_config_list = [1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.480775923,0.362512713,0.376398391,0.166316106,0.057218473, 0.306640625,0.482421875,0.4921875]
            elif condition == 'energy':
                activation_config_list = [0.476475343,0.254811264,0.196848913,0.078171897,0.020796094,0.182793935,0.312905312,0.181742668]

        if mode == 'SRE':
            prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.491372191,0.359013845,0.343125925,0.152990939,0.056478828, 0.35546875,0.396484375,0.25]
            elif condition == 'energy':
                activation_config_list = [0.485655824,0.062730183,0.162549231,0.052158928,0.016514578,0.220743815,0.229764938,0.121049881]

    elif cate.startswith('VGG8'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 2, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 2, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 2, 'stride': 2, 'padding': 0})

        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': 10, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        activation_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        if mode == 'shapePipe':  #
            prune_config_list = [1,0.4,0.45,0.4,0.45,0.4,0.45,1]
            if condition == 'latency':
                activation_config_list = [0.344686694,0.157210369,0.287748173,0.090109351,0.155040076,0.061958911,0.169097363,0.337890625]
            elif condition == 'energy':
                activation_config_list = [0.34022552,0.062373822,0.090715894,0.043433174,0.064760846,0.012919251,0.058685216,0.20734787]

        if mode == 'shape':  #
            prune_config_list = [1,0.4,0.45,0.4,0.45,0.4,0.45,1]
            if condition == 'latency':
                activation_config_list = [0.344686694,0.157210369,0.287748173,0.090109351,0.155040076,0.061958911,0.169097363,0.337890625]
            elif condition == 'energy':
                activation_config_list = [0.34022552,0.062373822,0.090715894,0.043433174,0.064760846,0.012919251,0.058685216,0.20734787]

        if mode == 'onlyRCP':
            prune_config_list = [1,0.4,0.45,0.4,0.45,0.4,0.45,1]
            if condition == 'latency':
                activation_config_list = [0.427267075,0.298784256,0.421813965,0.197250366,0.192016602,0.162994385,0.350585938, 0.318359375]
            elif condition == 'energy':
                activation_config_list = [0.422358407,0.138114118,0.142785206,0.097993309,0.08940535,0.046757746,0.121861951,0.215984344]

        if mode == 'SRE':  
            prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            if condition == 'latency':
                activation_config_list = [0.490655899,0.328275681,0.242111206,0.190559387,0.230377197,0.191131592,0.331054688, 0.251953125]
            elif condition == 'energy':
                activation_config_list = [0.466162434,0.158195103,0.143829882,0.107899851,0.124497255,0.073033346,0.102790303,0.161623001]

    elif cate.startswith('VGG16'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 2, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 2, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 2, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 2,'padding': 0})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})

        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1,'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 2,'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': 10, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0]
        activation_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0]

        if mode == 'shapePipe':  
            prune_config_list = [1,0.6,0.7,0.4,0.6,0.4,0.4,0.6,0.4,0.15,0.15,0.15,0.15,1]  # 2.26
            if condition == 'latency':
                activation_config_list = [0.36775251,0.267812092,0.200574517,0.127319516,0.225794037,0.12437569,0.029331598,0.205554199,0.127456445,0.185515747,0.286007129,0.205372852,0.13616875,0.4453125]
            elif condition == 'energy':
                activation_config_list = [0.363166345,0.148384679,0.121122564,0.071883987,0.096186837,0.058416726,0.012865088,0.046325258,0.025768053,0.039229295,0.101616043,0.068516384,0.036730649,0.265857697]

        if mode == 'shape': 
            prune_config_list = [1,0.6,0.7,0.4,0.6,0.4,0.4,0.6,0.4,0.15,0.15,0.15,0.15,1]  # 2.26
            if condition == 'latency':
                activation_config_list = [0.36775251,0.267812092,0.200574517,0.127319516,0.225794037,0.12437569,0.029331598,0.205554199,0.127456445,0.185515747,0.286007129,0.205372852,0.13616875,0.4453125]
            elif condition == 'energy':
                activation_config_list = [0.363166345,0.148384679,0.121122564,0.071883987,0.096186837,0.058416726,0.012865088,0.046325258,0.025768053,0.039229295,0.101616043,0.068516384,0.036730649,0.265857697]

        if mode == 'onlyRCP':
            prune_config_list = [1, 0.6, 0.7, 0.4, 0.6, 0.4, 0.4, 0.6, 0.4, 0.15, 0.15, 0.15, 0.15, 1]  # 2.26
            if condition == 'latency':
                activation_config_list = [0.427267075,0.38429451,0.268127441,0.246421814,0.251861572,0.172332764,0.156494141,0.219482422,0.198730469,0.322021484,0.397460938,0.287597656,0.192871094,0.39453125]
            elif condition == 'energy':
                activation_config_list = [0.422358407,0.226579683,0.171122051,0.152310502,0.131356741,0.088098522,0.076004293,0.069484733,0.040860786,0.078115678,0.156235578,0.097421621,0.074882486,0.216384888]

        if mode == 'SRE':  
            prune_config_list = [1 for i in range(14)]
            if condition == 'latency':
                activation_config_list = [0.490655899,0.343544006,0.290756226,0.234703064,0.23046875,0.141326904,0.118377686,0.241943359,0.202392578,0.26171875,0.349121094,0.265136719,0.232421875,0.3203125]
            elif condition == 'energy':
                activation_config_list = [0.466162434,0.182697809,0.180316846,0.130486382,0.11739458,0.070781681,0.056881799,0.085042,0.056904342,0.066696935,0.078417884,0.086909824,0.071731991,0.194137573]

    elif cate.startswith('NewResNet'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 7
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 8
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})

        # output
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 4, 'stride': 4, 'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        activation_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        if mode == 'shapePipe':
            prune_config_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
            if condition == 'latency':
                activation_config_list = [0.333930033,0.287247461,0.338563304,0.248332294,0.385053619,0.317804807,0.303915411,0.434151663,0.223029183,0.297872058,0.27897515,0.249021664,0.211495178,0.158595264,0.403295361,0.145091626,0.227050781,0.476344385,0.191708594,0.206635669,0.38671875]
            elif condition == 'energy':
                activation_config_list = [0.321292916,0.163024058,0.207756824,0.090257286,0.166531431,0.181332277,0.156701267,0.217247553,0.124963652,0.138852589,0.196042137,0.153003244,0.094860371,0.058948783,0.049464961,0.043504785,0.056263648,0.170812779,0.075549078,0.063187164,0.038780212]

        if mode == 'shape':  #
            prune_config_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
            if condition == 'latency':
                activation_config_list = [0.333930033,0.287247461,0.338563304,0.248332294,0.385053619,0.317804807,0.303915411,0.434151663,0.223029183,0.297872058,0.27897515,0.249021664,0.211495178,0.158595264,0.403295361,0.145091626,0.227050781,0.476344385,0.191708594,0.206635669,0.38671875]
            elif condition == 'energy':
                activation_config_list = [0.321292916,0.163024058,0.207756824,0.090257286,0.166531431,0.181332277,0.156701267,0.217247553,0.124963652,0.138852589,0.196042137,0.153003244,0.094860371,0.058948783,0.049464961,0.043504785,0.056263648,0.170812779,0.075549078,0.063187164,0.038780212]

        if mode == 'onlyRCP':
            prune_config_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
            if condition == 'latency':
                activation_config_list = [0.490655899,0.390270233,0.4277668,0.326807022,0.450920105,0.377319336,0.331977844,0.507080078,0.281234741,0.341087341,0.332794189,0.325714111,0.294677734,0.179962158,0.512268066,0.259521484,0.270629883,0.51171875,0.241088867,0.313232422,0.380859375]
            elif condition == 'energy':
                activation_config_list = [0.474194527,0.241285518,0.281529923,0.133467532,0.20608653,0.229398489,0.179889096,0.273577837,0.172555632,0.165154311,0.249156475,0.217527919,0.151993434,0.085051369,0.080707285,0.075148582,0.063432154,0.187733665,0.117119181,0.128815916,0.039531708]

        if mode == 'SRE':  
            prune_config_list = [1.0 for i in range(21)]
            if condition == 'latency':
                activation_config_list = [0.490655899,0.370023727,0.333230972,0.328590393,0.454175949,0.342330933,0.305587769,0.467735291,0.27960968,0.498626709,0.326751709,0.326751709,0.335632324,0.255432129,0.50604248,0.308105469,0.305053711,0.506347656,0.182373047,0.267211914,0.34765625]
            elif condition == 'energy':
                activation_config_list = [0.466162434,0.054989421,0.192344977,0.080609414,0.125777,0.168341517,0.133743776,0.170728538,0.122526738,0.129185915,0.192331791,0.174511009,0.104772382,0.067378561,0.080242833,0.085671425,0.071281274,0.082973586,0.013970455,0.100426992,0.038475037]
    else:
        assert 0, f'not support {cate}'

    if cate == 'AlexNet':
        input_params = {'activation_bit': 8, 'input_shape': (1, 3, 224, 224)}
    elif cate == 'ZFNet':
        input_params = {'activation_bit': 8, 'input_shape': (1, 3, 224, 224)}
    else:
        input_params = {'activation_bit': 8, 'input_shape': (1, 3, 32, 32)}


    for i in range(len(layer_config_list)):
        quantize_config_list.append({'weight_bit': 8, 'activation_bit': 8})
        if 'input_index' in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]['input_index'])
        else:
            input_index_list.append([-1])

    if cate != 'NewResNet':
        L = len(layer_config_list)
        for i in range(L-1, -1, -1):
            if layer_config_list[i]['type'] == 'conv':
                layer_config_list.insert(i+1, {'type': 'bn', 'features': layer_config_list[i]['out_channels']})
                quantize_config_list.insert(i+1, {'weight_bit': 8, 'activation_bit': 8})
                input_index_list.insert(i+1, [-1])
                for j in range(i + 2, len(layer_config_list), 1):
                    for relative_input_index in range(len(input_index_list[j])):
                        if j + input_index_list[j][relative_input_index] < i + 1:
                            input_index_list[j][relative_input_index] -= 1
    else:
        L = len(layer_config_list)
        for i in range(L - 1, -1, -1):
            if layer_config_list[i]['type'] == 'conv' and layer_config_list[i+1]['type'] == 'element_sum' and (layer_config_list[i]['out_channels']//layer_config_list[i]['in_channels']==2):
                layer_config_list.insert(i + 1, {'type': 'bn', 'features': layer_config_list[i]['out_channels']})
                quantize_config_list.insert(i + 1, {'weight_bit': 8, 'activation_bit': 8})
                input_index_list.insert(i + 1, [-1])
                for j in range(i + 2, len(layer_config_list), 1):
                    for relative_input_index in range(len(input_index_list[j])):
                        if j + input_index_list[j][relative_input_index] < i + 1:
                            input_index_list[j][relative_input_index] -= 1

    print(layer_config_list)
    print(quantize_config_list)
    print(input_index_list)

    # generate net
    net = NetworkGraph(hardware_config, layer_config_list, quantize_config_list, input_index_list, reuse_config_list, prune_config_list, input_params,activation_config_list)

    return net


if __name__ == '__main__':
    hardware_config = {'xbar_size': 128, 'input_bit': 8, 'weight_bit': 8, 'quantize_bit': 8}
    net = get_net(hardware_config, 'Vgg16', 10, 'naive')
    net.get_structure()
    net.get_weights()
    print(net.net_info)
