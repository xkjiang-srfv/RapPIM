#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
from Hardware_Model.PE import ProcessElement
from Hardware_Model.Buffer import buffer
from Interface.interface import *


class PE_latency_analysis:
    def __init__(self, SimConfig_path, max_row=0, max_column=0, inprecision=8, default_inbuf_size=16, default_outbuf_size=16, default_indexbuf_size=0):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # default_buf_size: default input buffer size (KB)
        PE_config = cp.ConfigParser()
        PE_config.read(SimConfig_path, encoding='UTF-8')
        self.inbuf_pe = buffer(default_buf_size=default_inbuf_size)
        self.outbuf_pe = buffer(default_buf_size=default_outbuf_size)
        self.indexbuf_pe = buffer(default_buf_size=default_indexbuf_size)
        self.PE = ProcessElement(SimConfig_path)
        self.digital_period = 1 / float(PE_config.get('Digital module', 'Digital_Frequency')) * 1e3  # unit: ns
        self.XBar_size = list(map(int, PE_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.OU_size = list(map(int, PE_config.get('Crossbar level', 'OU_Size').split(',')))
        self.max_row = max_row
        self.max_column = max_column
        self.inprecision = inprecision

        DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))
        ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))

        Row = self.XBar_size[0]
        Column = self.XBar_size[1]
        # ns  (using NVSim)
        decoderLatency_dict = {
            1: 0.27933  # 1:8, technology 65nm
        }
        decoder1_8 = decoderLatency_dict[1]
        Row_per_DAC = math.ceil(Row/DAC_num)
        m = 1
        while Row_per_DAC > 0:
            Row_per_DAC = Row_per_DAC // 8
            m += 1
        self.decoderLatency = m * decoder1_8

        # ns
        muxLatency_dict = {
            1: 32.744/1000
        }
        mux8_1 = muxLatency_dict[1]
        m = 1
        Column_per_ADC = math.ceil(Column / ADC_num)
        while Column_per_ADC > 0:
            Column_per_ADC = Column_per_ADC // 8
            m += 1
        self.muxLatency = m * mux8_1

        self.PE_buf_rlatency = 0
        self.PE_buf_wlatency = 0
        self.PE_inbuf_rlatency = 0
        self.PE_inbuf_wlatency = 0
        self.PE_outbuf_rlatency = 0
        self.PE_outbuf_wlatency = 0
        self.PE_indexbuf_rlatency = 0
        self.PE_indexbuf_wlatency = 0

        self.xbar_latency = 0
        self.DAC_latency = 0
        self.ADC_latency = 0
        self.iReg_latency = 0
        self.oReg_latency = 0
        self.input_demux_latency = 0
        self.output_mux_latency = 0
        self.shiftreg_latency = 0
        self.adder_latency = 0

        self.computing_latency = self.DAC_latency + self.xbar_latency + self.ADC_latency
        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + self.adder_latency + self.output_mux_latency + self.oReg_latency
        self.PE_buf_wlatency = self.PE_inbuf_wlatency + self.PE_outbuf_wlatency + self.PE_indexbuf_wlatency
        self.PE_buf_rlatency = self.PE_inbuf_rlatency + self.PE_outbuf_rlatency + self.PE_indexbuf_rlatency
        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency

    def update_PE_latency(self, OU_row, read_column, indata, rdata, wdata, mode,activation_ratio,layerType,condition):
        # update the latency computing when indata and rdata change
        multiple_time = math.ceil(self.inprecision / self.PE.DAC_precision) * OU_row * (read_column / self.OU_size[1])  * activation_ratio
        self.PE.calculate_DAC_latency()
        self.PE.calculate_xbar_read_latency()
        self.PE.calculate_ADC_latency()

        self.iReg_latency = multiple_time * self.digital_period * self.OU_size[0]
        self.input_demux_latency = multiple_time * self.decoderLatency  #
        self.DAC_latency = multiple_time * self.PE.DAC_latency
        self.xbar_latency = multiple_time * self.PE.xbar_read_latency
        self.output_mux_latency = multiple_time * self.muxLatency
        self.ADC_latency = multiple_time * self.PE.ADC_latency * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.shiftreg_latency = multiple_time * self.digital_period * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.adder_latency = multiple_time * self.digital_period * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.oReg_latency = OU_row * (read_column / self.OU_size[1]) * self.digital_period  

        self.computing_latency = self.DAC_latency + self.xbar_latency + self.ADC_latency    
        self.inbuf_pe.calculate_buf_write_latency(indata)  
        self.inbuf_pe.calculate_buf_read_latency(rdata)    

   
        self.PE_inbuf_rlatency = self.inbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num    
        # self.PE_inbuf_wlatency = self.inbuf_pe.buf_wlatency * math.ceil(self.max_row / self.OU_size[0]) 
        self.PE_inbuf_wlatency = self.inbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num    


        self.outbuf_pe.calculate_buf_read_latency(wdata)  
        self.outbuf_pe.calculate_buf_write_latency(wdata) 

        self.PE_outbuf_rlatency = self.outbuf_pe.buf_rlatency * math.ceil(self.max_column / self.OU_size[1]) * (self.OU_size[1]/8) * OU_row * self.PE.PE_xbar_num  
        self.PE_outbuf_wlatency = self.outbuf_pe.buf_wlatency * math.ceil(self.max_column / self.OU_size[1]) * (self.OU_size[1]/8) * OU_row * self.PE.PE_xbar_num 
        if layerType == 'conv':
            if condition == 'latency':
                if mode == 'onlyRCP':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/8  
                if mode == 'shape':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/4  
                if mode == 'shapePipe':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision) / 4  
            elif condition == 'energy':
                if mode == 'onlyRCP':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/8  
                elif mode == 'shape' or mode == 'shapePipe':
                    self.iReg_latency += self.iReg_latency / math.ceil(self.inprecision / self.PE.DAC_precision) / 4 
        self.adder_latency = self.adder_latency + OU_row * (self.max_column / self.OU_size[1]) * self.digital_period  
        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + self.adder_latency + self.output_mux_latency + self.oReg_latency
        self.PE_indexbuf_rlatency = 0
        self.PE_buf_wlatency = self.PE_inbuf_wlatency + self.PE_outbuf_wlatency + self.PE_indexbuf_wlatency
        self.PE_buf_rlatency = self.PE_inbuf_rlatency + self.PE_outbuf_rlatency + self.PE_indexbuf_rlatency
        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    _test = PE_latency_analysis(test_SimConfig_path, 256, 256, 8, 16, 2, 32)
    _test.update_PE_latency(OU_row=32, read_column=256, indata=1, rdata=8, wdata=1, mode='naive')
    print(_test.PE_latency)
