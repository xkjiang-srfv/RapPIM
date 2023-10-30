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
        # multiple是所有OU的工作次数
        multiple_time = math.ceil(self.inprecision / self.PE.DAC_precision) * OU_row * (read_column / self.OU_size[1])  * activation_ratio
        # 其实只有ADC、DAC、Crossbar的延时需要初始化一下，即需要读取一下SimConfig.ini文件，其余的都是digital部分，包括iReg/oReg/ShiftReg/Adder/inputmux/outputmux，这些的器件工作一次的延时都是固定的
        self.PE.calculate_DAC_latency()
        self.PE.calculate_xbar_read_latency()
        self.PE.calculate_ADC_latency()

        # self.iReg_latency = multiple_time * self.digital_period  # OU工作一次，iReg读一次数据，但是一次性只能读出8bit来，也就是只能满足OU_Size[0]=8，如果OU_Size[0]=16的话，就需要读2次了
        self.iReg_latency = multiple_time * self.digital_period * self.OU_size[0]
        self.input_demux_latency = multiple_time * self.decoderLatency  #
        self.DAC_latency = multiple_time * self.PE.DAC_latency
        self.xbar_latency = multiple_time * self.PE.xbar_read_latency
        self.output_mux_latency = multiple_time * self.muxLatency
        self.ADC_latency = multiple_time * self.PE.ADC_latency * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.shiftreg_latency = multiple_time * self.digital_period * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.adder_latency = multiple_time * self.digital_period * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)
        self.oReg_latency = OU_row * (read_column / self.OU_size[1]) * self.digital_period  # 所有OU工作一次，oReg记录一次数据

        self.computing_latency = self.DAC_latency + self.xbar_latency + self.ADC_latency    # 计算延迟是DAC+ADC+Crossbar的延时
        # indata=outdata，都是OU_Size[0](单位：byte)
        # 计算buf_pe的单次写入与单次读取的延时，读取的单位是OU_Size[0] Byte，再乘上读取的次数。
        self.inbuf_pe.calculate_buf_write_latency(indata)  # 先从tile中写入数据inbuf_pe，indata是当前滑窗中一个input_channel中数据的大小
        self.inbuf_pe.calculate_buf_read_latency(rdata)    # 再从PE的inbuf_pe中读出数据来，rdata是OU_Size[0]*inputbit/8，就是要填满一个OU要出来多少数据

        # 在计算的时候，是按照同时产生所有output channel中的一个数据为基础频次的，
        # 这里你可能有一个疑问，按照这个意思我是好多次，每次都只读一个OU_size[1]大小的数据？难道不能一次性多读一点嘛？这其实是等价的，因为在计算计算buf_pe的单次写入与单次读取的延时时考虑了数据大小，如果按照一次性多读取数据的思路，那么就是单次读写的数据量大一点，读取次数少一点而已
        self.PE_inbuf_rlatency = self.inbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num     # OU_row * self.PE.PE_xbar_num就是有多少个OU行，self.inbuf_pe.buf_rlatency指的是一次性读取OU[0]这么大小的数据需要多少时间，因此该式就是从pe_buf中读取数据并填满所有OU row需要多少时间
        # self.PE_inbuf_wlatency = self.inbuf_pe.buf_wlatency * math.ceil(self.max_row / self.OU_size[0])  # self.max_row / self.OU_size[0]就是一个crossbar上有多少个OU行这样的话Tile就能根据你有多少个OU行给当前的inbuf写入数据了
        self.PE_inbuf_wlatency = self.inbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num     # 我理解的PE_inbuf_wlatency应该与PE_inbuf_rlatency相同，我往外读出多少数据就往里读入多少数据


        self.outbuf_pe.calculate_buf_read_latency(wdata)  # 一次性从outbuf_pe中读取的数据大小为1Byte，即一个数据
        self.outbuf_pe.calculate_buf_write_latency(wdata) # 一次性往outbuf_pe中写入的数据大小为1Byte

        self.PE_outbuf_rlatency = self.outbuf_pe.buf_rlatency * math.ceil(self.max_column / self.OU_size[1]) * (self.OU_size[1]/8) * OU_row * self.PE.PE_xbar_num  # 从outbuf_pe中读取数据的次数，读取到tilebuf中去,(self.OU_size[1]/8)指的是OU计算一次能出来多少byte数据，毕竟是按照一次性写入1byte计算的单次写入延时
        self.PE_outbuf_wlatency = self.outbuf_pe.buf_wlatency * math.ceil(self.max_column / self.OU_size[1]) * (self.OU_size[1]/8) * OU_row * self.PE.PE_xbar_num  # 往outbuf_tile中写入数据的次数，看这个意思是OU中所有的列计算完一次写一次，
        # 目前位于PE层面，不同的crossbar并行操作，我读取一次iReg数据，就要读取一次ARMT的数据和LAMT的数据，用输入寄存器存储ARMT与LAMT，但是ADC、DAC的工作次数没有发生任何变化。ARMT与LAMT的写入延迟在model_latency函数中给出
        if layerType == 'conv':
            if condition == 'latency':
                if mode == 'onlyRCP':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/8  # 需要加上ARMT的读取延时，iReg工作一次那我的ARMT就要工作一次，但是我一次只需要读取1bit就行，而不是像iReg那样读取8bit数据，且我读取一次能让8bit的input全部受用
                if mode == 'shape':
                    # 所有的高位计算结果都是在oReg中暂存的，往jointmodule中传输的值都是当前层的每个输出数据是大于零还是小于等于零的一个标志位而已，这个应该在tile层中做，因为最终计算结果是在tile中合并的
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/4  # 需要加上LAMT和ARMT的读取延时
                if mode == 'shapePipe':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision) / 4  # 需要加上LAMT和ARMT的读取延时
            elif condition == 'energy':
                if mode == 'onlyRCP':
                    self.iReg_latency += self.iReg_latency/math.ceil(self.inprecision / self.PE.DAC_precision)/8  # 需要加上LAMT和ARMT的读取延时
                elif mode == 'shape' or mode == 'shapePipe':
                    self.iReg_latency += self.iReg_latency / math.ceil(self.inprecision / self.PE.DAC_precision) / 4 # 需要加上LAMT和ARMT的读取延时
        self.adder_latency = self.adder_latency + OU_row * (self.max_column / self.OU_size[1]) * self.digital_period  # 同列OU部分和结果累加
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
