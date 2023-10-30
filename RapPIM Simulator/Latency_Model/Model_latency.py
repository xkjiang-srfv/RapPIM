import pandas as pd
import configparser as cp
from Interface.interface import *
from Mapping_Model.Tile_connection_graph import TCG
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.Pooling_latency import pooling_latency_analysis
from Hardware_Model.Buffer import buffer
import math
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

def merge_interval(interval):
    if len(interval) == 0:
        return []
    result = []
    interval.sort()
    lower_bound = interval[0][0]
    upper_bound = interval[0][1]
    for index in range(1, len(interval)):
        if interval[index][0] > upper_bound:
            result.append([lower_bound, upper_bound])
            lower_bound = interval[index][0]
            upper_bound = interval[index][1]
        else:
            if interval[index][1] > upper_bound:
                upper_bound = interval[index][1]
    result.append([lower_bound, upper_bound])
    return result


class Model_latency:
    def __init__(self, SimConfig_path, TCG_mapping):
        self.SimConfig_path = SimConfig_path
        model_config = cp.ConfigParser()
        model_config.read(SimConfig_path, encoding='UTF-8')
        self.inter_tile_bandwidth = float(model_config.get('Tile level', 'Inter_Tile_Bandwidth'))  # 20Gps
        self.OU_size = list(map(int, model_config.get('Crossbar level', 'OU_Size').split(',')))


        self.graph = TCG_mapping
        # self.NetStruct = NetStruct
        # self.multiple = multiple

        self.begin_time = []
        self.finish_time = []
        self.layer_tile_latency = []
        self.occupancy = []

        self.buffer_r_latency = []
        self.buffer_w_latency = []
        self.inbuffer_latency = []
        self.outbuffer_latency = []
        self.buffer_latency = []
        self.DAC_latency = []
        self.xbar_latency = []
        self.ADC_latency = []
        self.iReg_latency = []
        self.oReg_latency = []
        self.input_demux_latency = []
        self.output_mux_latency = []
        self.shiftreg_latency = []
        self.adder_latency = []
        self.jointmodule_latency = []
        self.pooling_latency = []
        self.digital_latency = []

        self.computing_latency = []
        self.compute_interval = []
        self.intra_tile_latency = []
        self.inter_tile_latency = []
        self.tile_merge_latency = []
        self.tile_transfer_latency = []

        self.total_buffer_r_latency = []
        self.total_buffer_w_latency = []
        self.total_buffer_latency = []
        self.total_DAC_latency = []
        self.total_xbar_latency = []
        self.total_ADC_latency = []
        self.total_iReg_latency = []
        self.total_oReg_latency = []
        self.total_input_demux_latency = []
        self.total_output_mux_latency = []
        self.total_shiftreg_latency = []
        self.total_adder_latency = []
        self.total_jointmodule_latency = []
        self.total_pooling_latency = []
        self.total_digital_latency = []

        self.total_computing_latency = []
        self.total_intra_tile_latency = []
        self.total_inter_tile_latency = []
        self.total_tile_merge_latency = []
        self.total_tile_transfer_latency = []

        self.layer_type = []
        self.layer_split = []
        self.pre_max_time = 0
    
    def layer_latency_initial(self):
        self.begin_time.append([])
        self.finish_time.append([])

        self.buffer_r_latency.append([])
        self.buffer_w_latency.append([])
        self.inbuffer_latency.append([])
        self.outbuffer_latency.append([])
        self.buffer_latency.append([])
        self.DAC_latency.append([])
        self.xbar_latency.append([])
        self.ADC_latency.append([])
        self.iReg_latency.append([])
        self.oReg_latency.append([])
        self.input_demux_latency.append([])
        self.output_mux_latency.append([])
        self.shiftreg_latency.append([])
        self.adder_latency.append([])
        self.jointmodule_latency.append([])
        self.pooling_latency.append([])
        self.digital_latency.append([])

        self.compute_interval.append([])
        self.computing_latency.append([])
        self.intra_tile_latency.append([])
        self.inter_tile_latency.append([])
        self.tile_merge_latency.append([])
        self.tile_transfer_latency.append([])

    def pipe_result_update(self, layer_type='conv', begin_time=0, compute_time=0, layer_id=0, temp_tile_latency=None, temp_pooling_latency=None, merge_time=0, transfer_time=0, output_size=0):
        if layer_type == 'conv':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.tile_buf_rlatency + temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency + temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency + temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'fc':
            self.begin_time[layer_id] = output_size * [begin_time]
            self.finish_time[layer_id] = output_size * [compute_time]
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.tile_buf_wlatency + temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency+temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency+temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency + temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency + temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'pooling':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.inbuf_rlatency + temp_pooling_latency.outbuf_wlatency + temp_pooling_latency.outbuf_rlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(temp_pooling_latency.inbuf_rlatency + temp_pooling_latency.outbuf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.outbuf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(0)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)
            self.digital_latency[layer_id].append(0)
            self.pooling_latency[layer_id].append(temp_pooling_latency.digital_latency)

            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'element_sum':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.tile_buf_wlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)
            self.digital_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

    def model_latency_output(self, module_information=1, layer_information=1):
        print(' ')
        total_latency = [0] * len(self.begin_time)
        if layer_information:
            for i in range(len(self.begin_time)):
                print("Layer", i, " type:", self.graph.net[i][0][0]['type'])
                total_latency[i] = self.total_buffer_latency[i] + self.total_computing_latency[i] + self.total_digital_latency[i] + self.total_intra_tile_latency[i] + self.total_inter_tile_latency[i] + self.total_pooling_latency[i]
                if module_information:
                    print("Total latency of layer", i, ":", total_latency[i])
                    print("Buffer latency of layer", i, ":", self.total_buffer_latency[i], '(', "%.2f" % (100 * self.total_buffer_latency[i] / total_latency[i]), '%)')
                    print("     read buffer latency of layer", i, ":", self.total_buffer_r_latency[i], '(', "%.2f" % (100 * self.total_buffer_r_latency[i] / total_latency[i]), '%)')
                    print("     write buffer latency of layer", i, ":", self.total_buffer_w_latency[i], '(', "%.2f" % (100 * self.total_buffer_w_latency[i] / total_latency[i]), '%)')
                    print("Computing latency of layer", i, ":", self.total_computing_latency[i], '(', "%.2f" % (100 * self.total_computing_latency[i] / total_latency[i]), '%)')
                    print("     DAC latency of layer", i, ":", self.total_DAC_latency[i], '(', "%.2f" % (100 * self.total_DAC_latency[i] / total_latency[i]), '%)')
                    print("     ADC latency of layer", i, ":", self.total_ADC_latency[i], '(', "%.2f" % (100 * self.total_ADC_latency[i] / total_latency[i]), '%)')
                    print("     xbar latency of layer", i, ":", self.total_xbar_latency[i], '(', "%.2f" % (100 * self.total_xbar_latency[i] / total_latency[i]), '%)')
                    print("Digital part latency of layer", i, ":", self.total_digital_latency[i], '(', "%.2f" % (100 * self.total_digital_latency[i] / total_latency[i]), '%)')
                    print("     iReg latency of layer", i, ":", self.total_iReg_latency[i], '(', "%.2f" % (100 * self.total_iReg_latency[i] / total_latency[i]), '%)')
                    print("     oReg latency of layer", i, ":", self.total_oReg_latency[i], '(', "%.2f" % (100 * self.total_oReg_latency[i] / total_latency[i]), '%)')
                    print("     input demux latency of layer", i, ":", self.total_input_demux_latency[i], '(', "%.2f" % (100 * self.total_input_demux_latency[i] / total_latency[i]), '%)')
                    print("     output mux latency of layer", i, ":", self.total_output_mux_latency[i], '(', "%.2f" % (100 * self.total_output_mux_latency[i] / total_latency[i]), '%)')
                    print("     shiftreg latency of layer", i, ":", self.total_shiftreg_latency[i], '(', "%.2f" % (100 * self.total_shiftreg_latency[i] / total_latency[i]), '%)')
                    print("     adder latency of layer", i, ":", self.total_adder_latency[i], '(', "%.2f" % (100 * self.total_adder_latency[i] / total_latency[i]), '%)')
                    print("     Jointmodule latency of layer", i, ":", self.total_jointmodule_latency[i], '(', "%.2f" % (100 * self.total_jointmodule_latency[i] / total_latency[i]), '%)')
                    print("Pooling module latency of layer", i, ":", self.total_pooling_latency[i], '(', "%.2f" % (100 * self.total_pooling_latency[i] / total_latency[i]), '%)')
                    print("Intra tile communication latency of layer", i, ":", self.total_intra_tile_latency[i], '(', "%.2f" % (100 * self.total_intra_tile_latency[i] / total_latency[i]), '%)')
                    print("Inter tile communication latency of layer", i, ":", self.total_inter_tile_latency[i], '(', "%.2f" % (100 * self.total_inter_tile_latency[i] / total_latency[i]), '%)')
                    print("     One layer merge latency of layer", i, ":", self.total_tile_merge_latency[i], '(', "%.2f" % (100 * self.total_tile_merge_latency[i] / total_latency[i]), '%)')
                    print("     Inter tile transfer latency of layer", i, ":", self.total_tile_transfer_latency[i], '(', "%.2f" % (100 * self.total_tile_transfer_latency[i] / total_latency[i]), '%)')
                print('----------------------------------------------')
        print("Entire latency:", str(sum(total_latency)), "ns")

        return sum(total_latency)


    def calculate_model_latency(self, mode,condition):
        # fill in input data kernel size by kernel size (column direction)
        for layer_id in range(len(self.graph.net)):  
            layer_dict = self.graph.net[layer_id][0][0]
            if layer_id == 0:
                self.layer_latency_initial()
                output_size = list(map(int, layer_dict['Outputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['Outputbit'])

                read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path, max_row=self.graph.layer_tileinfo[layer_id]['max_row'], max_column=self.graph.layer_tileinfo[layer_id]['max_column'], inprecision=inputbit, PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'], default_outbuf_size_tile=self.graph.net[layer_id][0][0]['xk_outbuf_tile'], default_inbuf_size_pe=self.graph.net[layer_id][0][0]['xk_inbuf_pe'], default_outbuf_size_pe=self.graph.net[layer_id][0][0]['xk_outbuf_pe'],mode=mode,layerType=layer_dict['type'],condition=condition)
                temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=self.graph.net[layer_id][0][0]['Outputchannel']/8/self.graph.net[layer_id][0][0]['tile_number']*8)
                temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency  
                temp_tile_latency.tile_buf_wlatency = temp_tile_latency.tile_buf_rlatency

                merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])
                max_time = 0
                for i in range(int(output_size[0] / layer_dict['Multiple'])):
                    for j in range(int(output_size[1] / layer_dict['Multiple'])):
                        self.pre_max_time = max_time
                        # indata = kernelsize*kernelsize * inputbit / 8
                        indata = temp_tile_latency.OU_size[0] * inputbit / 8  
                        rdata = temp_tile_latency.OU_size[0] * inputbit / 8
                        wdata = outputbit / 8
                        temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode, self.graph.net[layer_id][0][0]['activation_ratio'],layerType=layer_dict['type'],condition=condition)

                        begin_time = self.pre_max_time
                        compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                        self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                        max_time = compute_time

            else:
                if layer_dict['type'] == 'conv':
                    self.layer_latency_initial()
                    output_size = list(map(int, layer_dict['Outputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['Outputbit'])
                    # the input channel number each PE processes
                    read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                    OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              max_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                              max_column=self.graph.layer_tileinfo[layer_id][
                                                                  'max_column'], inprecision=inputbit,
                                                              PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                              default_outbuf_size_tile=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_tile'],
                                                              default_inbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_inbuf_pe'],
                                                              default_outbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_pe'],
                                                              mode=layer_dict['mode'],layerType=layer_dict['type'],condition=condition)
                    temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=self.graph.net[layer_id][0][0]['Outputchannel'] / 8 / self.graph.net[layer_id][0][0]['tile_number']*8)
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency 
                    temp_tile_latency.tile_buf_wlatency = temp_tile_latency.tile_buf_rlatency
                    merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                    transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                    max_time = 0
                    for i in range(int(output_size[0] / layer_dict['Multiple'])):
                        for j in range(int(output_size[1] / layer_dict['Multiple'])):
                            self.pre_max_time = max_time
                            # indata = kernelsize * kernelsize* inputbit / 8         
                            indata = temp_tile_latency.OU_size[0] * inputbit / 8
                            rdata = temp_tile_latency.OU_size[0] * inputbit / 8     
                            wdata = outputbit / 8                                  
                            temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode, self.graph.net[layer_id][0][0]['activation_ratio'],layerType=layer_dict['type'],condition=condition)

                            temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                            max_prelayer_time = 0  # the maximum time of the required input data (in all input layers)
                            for idx in temp_Inputindex:
                                tmp_time = self.finish_time[layer_id + idx][-1]
                                if tmp_time > max_prelayer_time:
                                    max_prelayer_time = tmp_time
                            begin_time = max(max_prelayer_time, self.pre_max_time)
                            OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / temp_tile_latency.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                            compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                            self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                            max_time = compute_time

                else:
                    if layer_dict['type'] == 'fc':
                        self.layer_latency_initial()
                        output_size = int(layer_dict['Outputchannel'])
                        input_size = int(layer_dict['Inputchannel'])
                        self.layer_split.append([input_size])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['Outputbit'])

                        read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                        OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              max_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                              max_column=self.graph.layer_tileinfo[layer_id][
                                                                  'max_column'], inprecision=inputbit,
                                                              PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                              default_outbuf_size_tile=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_tile'],
                                                              default_inbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_inbuf_pe'],
                                                              default_outbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_pe'],mode=layer_dict['mode'],layerType=layer_dict['type'])
                        temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=self.graph.net[layer_id][0][0]['Outputchannel'] / 8 / self.graph.net[layer_id][0][0]['tile_number']*8)
                        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency  
                        temp_tile_latency.tile_buf_wlatency = temp_tile_latency.tile_buf_rlatency
                        merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                        transfer_time = int(output_size * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                        # indata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8  
                        indata = temp_tile_latency.OU_size[0] * inputbit / 8
                        rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                        wdata = outputbit / 8
                        temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode,self.graph.net[layer_id][0][0]['activation_ratio'],layerType=layer_dict['type'])
                        temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                        max_prelayer_time = 0
                        for idx in temp_Inputindex:
                            tmp_time = self.finish_time[layer_id+idx][-1]
                            if tmp_time > max_prelayer_time:
                                max_prelayer_time = tmp_time
                        begin_time = max_prelayer_time
                        OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / temp_tile_latency.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                        compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                        self.pipe_result_update(layer_type='fc', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time, output_size=output_size)

                    elif layer_dict['type'] == 'pooling':
                        self.layer_latency_initial()
                        output_size = list(map(int, layer_dict['Outputsize']))
                        input_size = list(map(int, layer_dict['Inputsize']))
                        self.layer_split.append([input_size[1]])
                        kernelsize = int(layer_dict['Kernelsize'])
                        stride = int(layer_dict['Stride'])
                        inputchannel = int(layer_dict['Inputchannel'])
                        outputchannel = int(layer_dict['Outputchannel'])
                        padding = int(layer_dict['Padding'])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['Outputbit'])
                        temp_pooling_latency = pooling_latency_analysis(indata=0, rdata=0, outprecision=outputbit,
                                                                        default_inbuf_size=self.graph.net[layer_id][0][0]['xk_inbuf_pe'],
                                                                        default_outbuf_size=self.graph.net[layer_id][0][0]['xk_outbuf_tile'], default_inchannel=inputchannel)
                        temp_pooling_latency.outbuf.calculate_buf_read_latency(rdata=(outputchannel * outputbit / 8))
                        temp_pooling_latency.outbuf_rlatency = temp_pooling_latency.outbuf.buf_rlatency
                        merge_time = temp_pooling_latency.outbuf_rlatency
                        transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                        self.pre_max_time = 0
                        for i in range(int(output_size[0] / layer_dict['Multiple'])):
                            for j in range(int(output_size[1] / layer_dict['Multiple'])):
                                indata = inputchannel * kernelsize * kernelsize * inputbit / 8  
                                rdata = inputchannel * kernelsize ** 2 * inputbit / 8  
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                max_prelayer_time = 0
                                for idx in temp_Inputindex:
                                    tmp_time = self.finish_time[layer_id + idx][-1]
                                    if tmp_time > max_prelayer_time:
                                        max_prelayer_time = tmp_time
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)

                    elif layer_dict['type'] == 'element_sum':
                        self.layer_latency_initial()
                        Inputindex_list = list(map(int, layer_dict['Inputindex']))
                        assert len(Inputindex_list) > 1, "the number of element_sum's previous layers must > 1"
                        previous_layer_dict = self.graph.net[layer_id + Inputindex_list[0]][0][0]
                        output_size = list(map(int, previous_layer_dict['Outputsize']))
                        inputchannel = int(previous_layer_dict['Outputchannel'])
                        outputchannel = int(previous_layer_dict['Outputchannel'])
                        inputbit = int(previous_layer_dict['Outputbit'])
                        outputbit = int(previous_layer_dict['Outputbit'])

                        merge_time = 0
                        transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth)

                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              max_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                              max_column=self.graph.layer_tileinfo[layer_id][
                                                                  'max_column'], inprecision=inputbit,
                                                              PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                              default_outbuf_size_tile=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_tile'],
                                                              default_inbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_inbuf_pe'],
                                                              default_outbuf_size_pe=self.graph.net[layer_id][0][0][
                                                                  'xk_outbuf_pe'],mode=layer_dict['mode'],layerType=layer_dict['type'])
                        temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=(len(Inputindex_list) * inputchannel * inputbit / 8))
                        temp_tile_latency.outbuf_tile.calculate_buf_write_latency(wdata=(inputchannel * inputbit / 8))
                        temp_tile_latency.adder_latency = temp_tile_latency.digital_period

                        self.pre_max_time = 0
                        for i in range(output_size[0]):
                            for j in range(output_size[1]):
                                max_prelayer_time = 0  # the maximum time of the required input data (in all input layers)
                                for idx in Inputindex_list:
                                    tmp_time = self.finish_time[layer_id + idx][-1]
                                    if tmp_time > max_prelayer_time:
                                        max_prelayer_time = tmp_time
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                compute_time = merge_time + transfer_time + temp_tile_latency.adder_latency + temp_tile_latency.outbuf_tile.buf_rlatency + temp_tile_latency.outbuf_tile.buf_wlatency + begin_time
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='element_sum', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)

            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1] - self.compute_interval[layer_id][l][0])

            self.occupancy.append(temp_runtime / (max(self.finish_time[layer_id]) - min(self.begin_time[layer_id])))
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_buffer_r_latency.append(sum(self.buffer_r_latency[layer_id]))
            self.total_buffer_w_latency.append(sum(self.buffer_w_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_iReg_latency.append(sum(self.iReg_latency[layer_id]))
            self.total_oReg_latency.append(sum(self.oReg_latency[layer_id]))
            self.total_input_demux_latency.append(sum(self.input_demux_latency[layer_id]))
            self.total_output_mux_latency.append(sum(self.output_mux_latency[layer_id]))
            self.total_shiftreg_latency.append(sum(self.shiftreg_latency[layer_id]))
            self.total_adder_latency.append(sum(self.adder_latency[layer_id]))
            self.total_jointmodule_latency.append(sum(self.jointmodule_latency[layer_id]))
            self.total_pooling_latency.append(sum(self.pooling_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_inter_tile_latency.append(sum(self.inter_tile_latency[layer_id]))
            self.total_intra_tile_latency.append(sum(self.intra_tile_latency[layer_id]))
            self.total_tile_merge_latency.append(sum(self.tile_merge_latency[layer_id]))
            self.total_tile_transfer_latency.append(sum(self.tile_transfer_latency[layer_id]))


def extraLatencyCal(model_latency,mode,graph,model_name,poolLatency):
    PE_config = cp.ConfigParser()
    PE_config.read("../SimConfig.ini", encoding='UTF-8')
    digital_period = 1 / float(PE_config.get('Digital module', 'Digital_Frequency')) * 1e3
    layerNum = 0
    adderNum = 16
    if(mode == 'onlyRCP'):
        for layer_id in range(len(graph.net)):
            layer_dict = graph.net[layer_id][0][0]
            if layer_dict['type'] == 'conv':
                 if(layerNum < len(RCPruneflag[model_name])-1):
                    RCPBitMap = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period / layer_dict['tile_number'] 
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 

                    Tilelatency = buffer()
                    Tilelatency.calculate_buf_read_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)
                    Tilelatency.calculate_buf_write_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)

                    Img2colLatency = activationDivision[model_name][layerNum+1][0] * activationDivision[model_name][layerNum+1][1]* digital_period
                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * 2 * digital_period   
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period   

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

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum + 1][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum + 1][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum+1][0] + activationDivision[model_name][layerNum+1][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][1] / 8 * digital_period
                    model_latency = model_latency +RCPBitMap_comlatency+RCPBitMap_wlatency +RCPBitMap_rlatency+Tilelatency.buf_rlatency + Tilelatency.buf_wlatency+Img2colLatency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency
                if(layerNum == 0):
                    Img2colLatency = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]* digital_period

                    RCPBitMap = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period 
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8 
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8 

                    if RCPruneflag[model_name][layerNum] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   

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

                    if RCPruneflag[model_name][layerNum] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum][0] + activationDivision[model_name][layerNum][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum][1] / 8 * digital_period

                    model_latency = model_latency+Img2colLatency+RCPBitMap_comlatency+RCPBitMap_rlatency+RCPBitMap_wlatency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency
                layerNum += 1
        return model_latency
    if (mode == 'shape'):
        for layer_id in range(len(graph.net)):
            layer_dict = graph.net[layer_id][0][0]
            if layer_dict['type'] == 'conv':
                if(layerNum < len(RCPruneflag[model_name])-1):
                    RCPBitMap = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0] 
                    RCPBitMap_comlatency = RCPBitMap*digital_period / layer_dict['tile_number'] 
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8 / layer_dict['tile_number'] 
                    Tilelatency = buffer()
                    Tilelatency.calculate_buf_read_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)
                    Tilelatency.calculate_buf_write_latency(layer_dict['Outputchannel'] * layer_dict['Outputsize'][0] * layer_dict['Outputsize'][1] / 8)

                    Img2colLatency = activationDivision[model_name][layerNum+1][0] * activationDivision[model_name][layerNum+1][1]* digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period  
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period  

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

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum + 1][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum + 1][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum+1][0] + activationDivision[model_name][layerNum+1][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][1] / 8 * digital_period
                    model_latency = model_latency +RCPBitMap_comlatency+RCPBitMap_wlatency +RCPBitMap_rlatency+Tilelatency.buf_rlatency + Tilelatency.buf_wlatency+Img2colLatency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency
                if(layerNum == 0):
                    Img2colLatency = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]* digital_period

                    RCPBitMap = activationDivision[model_name][layerNum][0] * activationDivision[model_name][layerNum][1]  
                    RCPBitMap_comlatency = RCPBitMap*digital_period  
                    RCPBitMap_wlatency = RCPBitMap * digital_period / 8  
                    RCPBitMap_rlatency = RCPBitMap * digital_period / 8  

                    if RCPruneflag[model_name][layerNum] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   

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

                    if RCPruneflag[model_name][layerNum] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        ARMT_wlatency = (activationDivision[model_name][layerNum][0] + activationDivision[model_name][layerNum][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum][1] / 8 * digital_period

                    model_latency = model_latency+Img2colLatency+RCPBitMap_comlatency+RCPBitMap_wlatency+RCPBitMap_rlatency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency
                LAMT_write_latency = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0]/8*digital_period
                model_latency += LAMT_write_latency
                layerNum += 1
        return model_latency
    elif (mode == 'shapePipe'):
        for layer_id in range(len(graph.net)):
            layer_dict = graph.net[layer_id][0][0]
            if layer_dict['type'] == 'conv':
                if(layerNum < len(RCPruneflag[model_name])-1):
                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * 2 * digital_period  
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum+1][0]-1)*(activationDivision[model_name][layerNum+1][1]-1) * digital_period  
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

                    if RCPruneflag[model_name][layerNum+1] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum + 1][0]+activationDivision[model_name][layerNum + 1][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum + 1][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum + 1][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum+1][0] + activationDivision[model_name][layerNum+1][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1: 
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum + 1][1] / 8 * digital_period
                    model_latency = model_latency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency

                if(layerNum == 0):
                    if RCPruneflag[model_name][layerNum] == 3:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * 2 * digital_period 
                    elif RCPruneflag[model_name][layerNum] == 1:  
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period    
                    else:
                        AdderLatency = (activationDivision[model_name][layerNum][0]-1)*(activationDivision[model_name][layerNum][1]-1) * digital_period   

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

                    if RCPruneflag[model_name][layerNum] == 3: 
                        CompareLatency = (activationDivision[model_name][layerNum][0]+activationDivision[model_name][layerNum][1])*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:  
                        CompareLatency = activationDivision[model_name][layerNum][0]*digital_period
                    else:
                        CompareLatency = activationDivision[model_name][layerNum][1]*digital_period

                    if RCPruneflag[model_name][layerNum+1] == 3:  
                        ARMT_wlatency = (activationDivision[model_name][layerNum][0] + activationDivision[model_name][layerNum][1])/8*digital_period
                    elif RCPruneflag[model_name][layerNum+1] == 1:
                        ARMT_wlatency = activationDivision[model_name][layerNum][0]/8 * digital_period
                    else:
                        ARMT_wlatency = activationDivision[model_name][layerNum][1] / 8 * digital_period

                    model_latency = model_latency+AdderLatency+sortLatency+CompareLatency+ARMT_wlatency
                LAMT_write_latency = layer_dict['Outputchannel']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][0]/8*digital_period
                model_latency += LAMT_write_latency
                layerNum += 1
        return model_latency
    return model_latency


if __name__ == '__main__':
    model_name = ['AlexNet','ZFNet','VGG8','VGG16','NewResNet']
    condition = 'latency'
    mode = ['SRE','onlyRCP','shape','shapePipe']
    # mode = ['shape']
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    result = pd.DataFrame() 
    performance_analyse = pd.DataFrame() 
    model_latency = [0.0] * len(mode)  
    buffer_latency = [0.0] * len(mode)  
    dac_latency = [0.0] * len(mode) 
    crossbar_latency = [0.0] * len(mode)  
    adc_latency = [0.0] * len(mode)  
    shiftreg_latency = [0.0] * len(mode)  
    adder_latency = [0.0] * len(mode)  
    latency_speedup = [0.0] * len(mode)  
    model_baseline = 0

    for i in range(0, len(model_name)):
        for j in range(0, len(mode)):
            __TestInterface = TrainTestInterface(model_name[i], 'MNSIM.Interface.cifar10', test_SimConfig_path, mode[j],condition)
            structure_file = __TestInterface.get_structure()
            __TCG_mapping = TCG(structure_file, test_SimConfig_path)    
            __latency = Model_latency(test_SimConfig_path, __TCG_mapping)
            __latency.calculate_model_latency(mode[j],condition)
            model_latency[j] = __latency.model_latency_output(1, 1)
            model_latency[j] = extraLatencyCal(model_latency[j],mode[j],__TCG_mapping,model_name[i],poolLatency=__latency.pooling_latency)
            if mode[j] == 'SRE':
                model_baseline = model_latency[j]
            latency_speedup[j] = model_baseline / model_latency[j]

        # result[model_name[i] + '_model_latency'] = model_latency
        result[model_name[i] + '_latency_speedup'] = latency_speedup

    result.to_csv('latency_info' + '.csv')