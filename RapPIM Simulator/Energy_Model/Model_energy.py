#!/usr/bin/python
# -*-coding:utf-8-*-
import pandas as pd
import configparser as cp
from Interface.interface import *
from Mapping_Model.Tile_connection_graph import TCG
from Power_Model.Model_inference_power import Model_inference_power
from Latency_Model.Model_latency import Model_latency
from Hardware_Model.Buffer import buffer
activationDivision = {
    "AlexNet":[(363,3025),(2400,729),(2304,169),(3456,169),(3456,169),(0,0)],
    "ZFNet":[(147,12100),(1200,676),(1152,169),(1728,169),(1728,169),(0,0)],
    "VGG8":[(27,1024),(1152,1024),(1152,256),(2304,256),(2304,64),(4608,64),(4608,4),(0,0)],
    "VGG16":[(27,1024),(576,1024),(576,256),(1152,256),(1152,64),(2304,64),(2304,64),(2304,16),(4608,16),(4608,16),(4608,4),(4608,4),(4608,4),(0,0)],
    "NewResNet":[(27,1024),(576,1024),(576,1024),(576,1024),(576,1024),(64,256),(576,256),(1152,256),(1152,256),(1152,256),(128,64),(1152,64),(2304,64),(2304,64),(2304,64),(256,16),(2304,16),(4608,16),(4608,16),(4608,16),(0,0)]
}

RCPruneflag = {
    "AlexNet":[2,3,1,1,1],
    "ZFNet":[2,1,1,1,1],
    "VGG8":[2,3,1,1,1,1,1],
    "VGG16":[2,1,1,1,1,1,1,1,1,1,1,1,1],
    "NewResNet":[1 for i in range(20)]
}

class Model_energy:
    def __init__(self, NetStruct, SimConfig_path, multiple, TCG_mapping, mode,condition):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        modelL_config = cp.ConfigParser()
        modelL_config.read(self.SimConfig_path, encoding='UTF-8')
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num

        self.model_latency = Model_latency(SimConfig_path, TCG_mapping)
        self.model_latency.calculate_model_latency(mode,condition)
        self.model_power = Model_inference_power(NetStruct, SimConfig_path, multiple, TCG_mapping, mode)

        self.arch_energy = self.total_layer_num * [0]
        self.arch_xbar_energy = self.total_layer_num * [0]
        self.arch_ADC_energy = self.total_layer_num * [0]
        self.arch_DAC_energy = self.total_layer_num * [0]
        self.arch_digital_energy = self.total_layer_num * [0]
        self.arch_adder_energy = self.total_layer_num * [0]
        self.arch_shiftreg_energy = self.total_layer_num * [0]
        self.arch_iReg_energy = self.total_layer_num * [0]
        self.arch_oReg_energy = self.total_layer_num * [0]
        self.arch_input_demux_energy = self.total_layer_num * [0]
        self.arch_output_mux_energy = self.total_layer_num * [0]
        self.arch_jointmodule_energy = self.total_layer_num * [0]
        self.arch_buf_energy = self.total_layer_num * [0]
        self.arch_buf_r_energy = self.total_layer_num * [0]
        self.arch_buf_w_energy = self.total_layer_num * [0]
        self.arch_pooling_energy = self.total_layer_num * [0]

        self.arch_total_energy = 0
        self.arch_total_xbar_energy = 0
        self.arch_total_ADC_energy = 0
        self.arch_total_DAC_energy = 0
        self.arch_total_digital_energy = 0
        self.arch_total_adder_energy = 0
        self.arch_total_shiftreg_energy = 0
        self.arch_total_iReg_energy = 0
        self.arch_total_input_demux_energy = 0
        self.arch_total_jointmodule_energy = 0
        self.arch_total_buf_energy = 0
        self.arch_total_buf_r_energy = 0
        self.arch_total_buf_w_energy = 0
        self.arch_total_output_mux_energy = 0
        self.arch_total_pooling_energy = 0

        self.calculate_model_energy()

    def calculate_model_energy(self):
        for i in range(self.total_layer_num):
            self.arch_xbar_energy[i] = self.model_power.arch_xbar_power[i] * self.model_latency.total_xbar_latency[i]
            self.arch_ADC_energy[i] = self.model_power.arch_ADC_power[i] * self.model_latency.total_ADC_latency[i]
            self.arch_DAC_energy[i] = self.model_power.arch_DAC_power[i] * self.model_latency.total_DAC_latency[i]
            self.arch_adder_energy[i] = self.model_power.arch_adder_power[i] * self.model_latency.total_adder_latency[i]
            self.arch_shiftreg_energy[i] = self.model_power.arch_shiftreg_power[i] * self.model_latency.total_shiftreg_latency[i]
            self.arch_iReg_energy[i] = self.model_power.arch_iReg_power[i] * self.model_latency.total_iReg_latency[i]
            self.arch_oReg_energy[i] = self.model_power.arch_oReg_power[i] * self.model_latency.total_oReg_latency[i]
            self.arch_input_demux_energy[i] = self.model_power.arch_input_demux_power[i] * self.model_latency.total_input_demux_latency[i]
            self.arch_output_mux_energy[i] = self.model_power.arch_output_mux_power[i] * self.model_latency.total_output_mux_latency[i]
            self.arch_jointmodule_energy[i] = self.model_power.arch_jointmodule_power[i] * self.model_latency.total_jointmodule_latency[i]
            self.arch_buf_r_energy[i] = self.model_power.arch_buf_r_power[i] * self.model_latency.total_buffer_r_latency[i]
            self.arch_buf_w_energy[i] = self.model_power.arch_buf_w_power[i] * self.model_latency.total_buffer_w_latency[i]
            self.arch_buf_energy[i] = self.arch_buf_r_energy[i] + self.arch_buf_w_energy[i]
            self.arch_pooling_energy[i] = self.model_power.arch_pooling_power[i] * self.model_latency.total_pooling_latency[i]
            self.arch_digital_energy[i] = self.arch_shiftreg_energy[i] + self.arch_iReg_energy[i] + self.arch_oReg_energy[i] + self.arch_input_demux_energy[i] + self.arch_output_mux_energy[i] + self.arch_jointmodule_energy[i]
            self.arch_energy[i] = self.arch_xbar_energy[i] + self.arch_ADC_energy[i] + self.arch_DAC_energy[i] + self.arch_digital_energy[i] + self.arch_buf_energy[i] + self.arch_pooling_energy[i]

        self.arch_total_energy = sum(self.arch_energy)
        self.arch_total_xbar_energy = sum(self.arch_xbar_energy)
        self.arch_total_ADC_energy = sum(self.arch_ADC_energy)
        self.arch_total_DAC_energy = sum(self.arch_DAC_energy)
        self.arch_total_digital_energy = sum(self.arch_digital_energy)
        self.arch_total_adder_energy = sum(self.arch_adder_energy)
        self.arch_total_shiftreg_energy = sum(self.arch_shiftreg_energy)
        self.arch_total_iReg_energy = sum(self.arch_iReg_energy)
        self.arch_total_input_demux_energy = sum(self.arch_input_demux_energy)
        self.arch_total_output_mux_energy = sum(self.arch_output_mux_energy)
        self.arch_total_jointmodule_energy = sum(self.arch_jointmodule_energy)
        self.arch_total_buf_energy = sum(self.arch_buf_energy)
        self.arch_total_buf_r_energy = sum(self.arch_buf_r_energy)
        self.arch_total_buf_w_energy = sum(self.arch_buf_w_energy)
        self.arch_total_pooling_energy = sum(self.arch_pooling_energy)

    def model_energy_output(self, module_information=1, layer_information=1):
        print("Hardware energy:", self.arch_total_energy, "nJ")
        if module_information:
            print("		crossbar energy:", self.arch_total_xbar_energy, "nJ")
            print("		DAC energy:", self.arch_total_DAC_energy, "nJ")
            print("		ADC energy:", self.arch_total_ADC_energy, "nJ")
            print("		Buffer energy:", self.arch_total_buf_energy, "nJ")
            print("			|---read buffer energy:", self.arch_total_buf_r_energy, "nJ")
            print("			|---write buffer energy:", self.arch_total_buf_w_energy, "nJ")
            print("		Pooling energy:", self.arch_total_pooling_energy, "nJ")
            print("		Other digital part energy:", self.arch_total_digital_energy, "nJ")
            print("			|---adder energy:", self.arch_total_adder_energy, "nJ")
            print("			|---output-shift-reg energy:", self.arch_total_shiftreg_energy, "nJ")
            print("			|---input-reg energy:", self.arch_total_iReg_energy, "nJ")
            print("			|---input_demux energy:", self.arch_total_input_demux_energy, "nJ")
            print("			|---output_mux energy:", self.arch_total_output_mux_energy, "nJ")
            print("			|---joint_module energy:", self.arch_total_jointmodule_energy, "nJ")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                print("     Hardware energy:", self.arch_energy[i], "nJ")

        return self.arch_total_energy

def extraEnergy(modelEnergy,mode,graph,model_name):
    PE_config = cp.ConfigParser()
    PE_config.read("../SimConfig.ini", encoding='UTF-8')
    digital_period = 1 / float(PE_config.get('Digital module', 'Digital_Frequency')) * 1e3
    layerNum = 0
    if(mode == 'onlyRCP'):
        for layer_id in range(len(graph.net)):
            layer_dict = graph.net[layer_id][0][0]
            if layer_dict['type'] == 'conv':
                if(layerNum < len(RCPruneflag[model_name])-1):
                    RCPBitMap = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period / layer_dict['tile_number'] 
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 

                    RCPBitMap_comPower = 0.05 * 1e-3
                    RCPBitMap_Power = 0.23 * 1e-3
                    RCPBitMap_energy = (RCPBitMap_comlatency) * RCPBitMap_comPower + (RCPBitMap_wlatency + RCPBitMap_rlatency)*RCPBitMap_Power

                    Tilelatency = buffer()
                    Tilelatency.calculate_buf_read_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)
                    Tilelatency.calculate_buf_write_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)

                    TilebufPower = (layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8 / 64 / 1024) * 20.7 * 1e-3
                    TilebufRead_energy = Tilelatency.buf_rlatency * TilebufPower
                    TilebufWrite_energy = Tilelatency.buf_wlatency * TilebufPower

                    Img2colLatency = activationDivision[model_name][layerNum+1][0] * activationDivision[model_name][layerNum+1][1]* digital_period
                    Img2colPower = 0.23 * 1e-3
                    Img2col_energy = Img2colLatency * Img2colPower

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period  
                    AdderPower = 0.05 * 1e-3
                    Adder_energy = AdderLatency * AdderPower


                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        sumRC = activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1]
                        if sumRC!=0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        sumRC = activationDivision[model_name][layerNum + 1][0]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    else:
                        sumRC = activationDivision[model_name][layerNum + 1][1]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    sortPower = 0.05 * 1e-3
                    sort_energy = sortPower * sortLatency

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum + 1][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum + 1][1]*digital_period

                    ComparePower = 0.05 * 1e-3
                    Compare_energy = ComparePower * CompareLatency


                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum+1][0] + activationDivision[model_name][layerNum+1][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][1] / 8 * digital_period
                    ARMT_power = 0.23 * 1e-3
                    ARMT_energy = ARMT_power * ARMT_wlatency
                    modelEnergy += RCPBitMap_energy + TilebufRead_energy + TilebufWrite_energy + Img2col_energy + Adder_energy + sort_energy + Compare_energy + ARMT_energy
                if(layerNum == 0):
                    Img2colLatency = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]* digital_period
                    Img2colPower = 0.23 * 1e-3
                    Img2col_energy = Img2colLatency * Img2colPower

                    RCPBitMap = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period  
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8  
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8  
                    RCPBitMap_comPower = 0.05 * 1e-3
                    RCPBitMap_Power = 0.23 * 1e-3
                    RCPBitMap_energy = (RCPBitMap_comlatency) * RCPBitMap_comPower + (RCPBitMap_wlatency + RCPBitMap_rlatency)*RCPBitMap_Power


                    if RCPruneflag[model_name][layerNum] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   

                    AdderPower = 0.05 * 1e-3
                    Adder_energy = AdderLatency * AdderPower
                    if RCPruneflag[model_name][layerNum] == 3: 
                        sumRC = activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1]
                        if sumRC!=0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        sumRC = activationDivision[model_name][layerNum][0]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    else:
                        sumRC = activationDivision[model_name][layerNum][1]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period

                    sortPower = 0.05 * 1e-3
                    sort_energy = sortPower * sortLatency

                    if RCPruneflag[model_name][layerNum] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum][1]*digital_period

                    ComparePower = 0.05 * 1e-3
                    Compare_energy = ComparePower * CompareLatency

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum][0] + activationDivision[model_name][layerNum][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum][1] / 8 * digital_period
                    ARMT_power = 0.23 * 1e-3
                    ARMT_energy = ARMT_power * ARMT_wlatency
                    modelEnergy += Img2col_energy + RCPBitMap_energy + Adder_energy + sort_energy + Compare_energy + ARMT_energy
                layerNum += 1
        return modelEnergy
    if (mode == 'shape' or mode == 'shapePipe'):
        for layer_id in range(len(graph.net)):
            layer_dict = graph.net[layer_id][0][0]
            if layer_dict['type'] == 'conv':
                if(layerNum < len(RCPruneflag[model_name])-1):
                    RCPBitMap = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period / layer_dict['tile_number'] 
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 

                    RCPBitMap_comPower = 0.05 * 1e-3
                    RCPBitMap_Power = 0.23 * 1e-3
                    RCPBitMap_energy = (RCPBitMap_comlatency) * RCPBitMap_comPower + (RCPBitMap_wlatency + RCPBitMap_rlatency)*RCPBitMap_Power

                    Tilelatency = buffer()
                    Tilelatency.calculate_buf_read_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)
                    Tilelatency.calculate_buf_write_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)

                    TilebufPower = (layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8 / 64 / 1024) * 20.7 * 1e-3
                    TilebufRead_energy = Tilelatency.buf_rlatency * TilebufPower
                    TilebufWrite_energy = Tilelatency.buf_wlatency * TilebufPower

                    Img2colLatency = activationDivision[model_name][layerNum+1][0] * activationDivision[model_name][layerNum+1][1]* digital_period
                    Img2colPower = 0.23 * 1e-3
                    Img2col_energy = Img2colLatency * Img2colPower

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * 2 * digital_period   
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period  
                    AdderPower = 0.05 * 1e-3
                    Adder_energy = AdderLatency * AdderPower


                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        sumRC = activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1]
                        if sumRC!=0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        sumRC = activationDivision[model_name][layerNum + 1][0]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    else:
                        sumRC = activationDivision[model_name][layerNum + 1][1]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    sortPower = 0.05 * 1e-3
                    sort_energy = sortPower * sortLatency

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum + 1][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum + 1][1]*digital_period

                    ComparePower = 0.05 * 1e-3
                    Compare_energy = ComparePower * CompareLatency


                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        ARMT_wlatency = (activationDivision[model_name][layerNum+1][0] + activationDivision[model_name][layerNum+1][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][1] / 8 * digital_period
                    ARMT_power = 0.23 * 1e-3
                    ARMT_energy = ARMT_power * ARMT_wlatency
                    modelEnergy += RCPBitMap_energy + TilebufRead_energy + TilebufWrite_energy + Img2col_energy + Adder_energy + sort_energy + Compare_energy + ARMT_energy
                if(layerNum == 0):
                    Img2colLatency = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]* digital_period
                    Img2colPower = 0.23 * 1e-3
                    Img2col_energy = Img2colLatency * Img2colPower

                    RCPBitMap = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period  
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8  
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8  
                    RCPBitMap_comPower = 0.05 * 1e-3
                    RCPBitMap_Power = 0.23 * 1e-3
                    RCPBitMap_energy = (RCPBitMap_comlatency) * RCPBitMap_comPower + (RCPBitMap_wlatency + RCPBitMap_rlatency)*RCPBitMap_Power

                    if RCPruneflag[model_name][layerNum] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   

                    AdderPower = 0.05 * 1e-3
                    Adder_energy = AdderLatency * AdderPower
                    if RCPruneflag[model_name][layerNum] == 3: 
                        sumRC = activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1]
                        if sumRC!=0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        sumRC = activationDivision[model_name][layerNum][0]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period
                    else:
                        sumRC = activationDivision[model_name][layerNum][1]
                        if sumRC != 0:
                            sortLatency = sumRC * math.log2(sumRC) * digital_period

                    sortPower = 0.05 * 1e-3
                    sort_energy = sortPower * sortLatency

                    if RCPruneflag[model_name][layerNum] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum][1]*digital_period

                    ComparePower = 0.05 * 1e-3
                    Compare_energy = ComparePower * CompareLatency

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum][0] + activationDivision[model_name][layerNum][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum][1] / 8 * digital_period
                    ARMT_power = 0.23 * 1e-3
                    ARMT_energy = ARMT_power * ARMT_wlatency
                    modelEnergy += Img2col_energy + RCPBitMap_energy + Adder_energy + sort_energy + Compare_energy + ARMT_energy
                LAMT_write_latency = layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][0] / 8 * digital_period
                LAMT_power = 0.23 * 1e-3
                LAMT_energy = LAMT_power * LAMT_write_latency
                modelEnergy += LAMT_energy
                layerNum += 1
        return modelEnergy

    return modelEnergy




if __name__ == '__main__':
    model_name = ['AlexNet','ZFNet','VGG8','VGG16','NewResNet']
    # model_name = ['AlexNet']
    mode = [ 'SRE','onlyRCP', 'shape','shapePipe']
    # mode = ['onlyRCP']
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    condition = 'energy'
    result = pd.DataFrame()  
    model_energy = [0.0] * len(mode)  
    energy_efficiency = [0.0] * len(mode)  
    energy_improvement = [0.0] * len(mode)  
    energy_reduction = [0.0] * len(mode)
    model_baseline = 0

    for i in range(0, len(model_name)):
        for j in range(0, len(mode)):
            __TestInterface = TrainTestInterface(model_name[i], 'MNSIM.Interface.cifar10', test_SimConfig_path, mode[j],condition)
            structure_file = __TestInterface.get_structure()
            __TCG_mapping = TCG(structure_file, test_SimConfig_path)
            __energy = Model_energy(structure_file, test_SimConfig_path, __TCG_mapping.multiple, __TCG_mapping, mode[j],condition)
            model_energy[j] = __energy.model_energy_output(1, 1)
            model_energy[j] = extraEnergy(model_energy[j],mode[j],__TCG_mapping,model_name[i])

            if mode[j] == 'SRE':
                model_baseline = model_energy[j]
            energy_reduction[j] = (model_energy[j])/model_baseline

        result[model_name[i] + 'energy_reduction'] = energy_reduction

        result.to_csv('energy_info' + '.csv')

