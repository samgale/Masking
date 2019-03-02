# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 10:28:40 2016

@author: samg

modified version of iodaq.py from aibstim package
"""

from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
import numpy as np


def GetDevices():
    bufferSize = 1024 #set max buffer size
    devicenames = " "*bufferSize #build device string
    DAQmxGetSysDevNames(devicenames, bufferSize) #fill string with names
    return devicenames.strip().strip('\x00').split(', ')  #strip off null char for each
    
def GetAIChannels(device):
    bufferSize = 1024
    channels = " "*bufferSize
    DAQmxGetDevAIPhysicalChans(device,channels,bufferSize)
    return channels.strip().strip('\x00').split(', ')
    
def GetAOChannels(device):
    bufferSize = 1024
    channels = " "*bufferSize
    DAQmxGetDevAOPhysicalChans(device,channels,bufferSize)
    return channels.strip().strip('\x00').split(', ')

def GetDOPorts(device):
    bufferSize = 1024
    ports = " "*bufferSize
    DAQmxGetDevDOPorts(device, ports, bufferSize)
    return ports.strip().strip('\x00').split(', ')

def GetDIPorts(device):
    bufferSize = 1024
    ports = " "*bufferSize
    DAQmxGetDevDIPorts(device, ports, bufferSize)
    return ports.strip().strip('\x00').split(', ')

def GetDOLines(device):
    bufferSize = 1024
    lines = " "*bufferSize
    DAQmxGetDevDOLines(device, lines, bufferSize)
    return lines.strip().strip('\x00').split(', ')
    
def GetDILines(device):
    bufferSize = 1024
    lines = " "*bufferSize
    DAQmxGetDevDILines(device, lines, bufferSize)
    return lines.strip().strip('\x00').split(', ')


class AnalogInput(Task):
    '''
    create an AnalogInput object for a single channel
    
    example 1:
    
    import nidaq
    ai = nidaq.AnalogInput(device='Dev1',channel=0)
    ai = di.StartTask()
    ai.data # single data buffer
    ai.StopTask()
    ai.ClearTask()
    
    example 2:
    
    import nidaq
    ai = nidaq.AnalogInput(device='Dev1',channel=0,accumulate=True)
    ai = di.StartTask()
    ai.StopTask()
    ai.dataArray # array of data buffers
    ai.bufferCount
    ai.ClearTask()
    '''
    
    def __init__(self, 
                device='Dev1', 
                channel=0,
                voltageRange=(-10.0,10.0),
                sampRate=1000.0,
                bufferSize=1000,
                accumulate=False):
        
        # construct task
        Task.__init__(self)
        
        # set up task properties
        self.channel = channel
        self.voltageRange = voltageRange
        self.sampRate = sampRate
        self.bufferSize = bufferSize
        self.timeout = 10.0
        self.data = np.zeros(self.bufferSize,dtype=np.float64)
        self.accumulate = accumulate
        self.dataArray = []
        self.bufferCount = 0
        
        self.CreateAIVoltageChan(device+'/ai'+str(channel),'',DAQmx_Val_RSE,voltageRange[0],voltageRange[1],DAQmx_Val_Volts,None)

        self.CfgSampClkTiming('',sampRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,bufferSize)        
        
        # set up data buffer callback
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.bufferSize,0)        
        
        # set up task complete callback (unused since this class takes continuous samples)
        self.AutoRegisterDoneEvent(0)
        
    def EveryNCallback(self):
        try:
            read = int32()
            self.ReadAnalogF64(self.bufferSize,self.timeout,DAQmx_Val_Auto,self.data,self.bufferSize,byref(read),None)
            if self.accumulate:
                # append each data buffer to a list
                self.dataArray.extend(self.data.astype(np.float64).tolist())
            self.bufferCount+=1
        except Exception, e:
            print 'Failed to read buffer'
            
    def DoneCallback(self, status):
        print 'Status',status
        return 0 # the function should return an integer


class AnalogOutput(Task):
    '''
    create an AnalogOutput object for a single channel
    for unknown reason output rate is 1600 samples/s
    '''
    
    def __init__(self, 
                device='Dev1', 
                channel=0,
                voltageRange=(0.0,5.0)):
        
        # construct task
        Task.__init__(self)
        
        # set up task properties
        self.channel = channel
        self.voltageRange = voltageRange
        self.sampRate = 1600
        self.lastOut = np.nan

        self.CreateAOVoltageChan(device+'/ao'+str(channel),"",voltageRange[0],voltageRange[1],DAQmx_Val_Volts,None)

        self.AutoRegisterDoneEvent(0)

    def Write(self,values):
        '''Writes a numpy array of float64's to the analog output'''
        self.lastOut = values[-1]
        self.WriteAnalogF64(len(values),0,-1,DAQmx_Val_GroupByChannel,values,None,None)

    def DoneCallback(self,status):
        return 0


class DigitalInputs(Task):
    '''
    create a DigitalInputs object for all lines on a port
    
    example:
    
    import nidaq
    di = nidaq.DigitalInputs(device='Dev1',port=0)
    di = di.StartTask()
    data = di.Read()
    ch0 = data[0]
    di.StopTask()
    di.ClearTask()
    '''

    def __init__(self, 
                device='Dev1',
                port=0):
        
        # construct task
        Task.__init__(self)
        
        # set up task properties
        self.port = port
        lines = GetDILines(device)
        lines = [l for l in lines if 'port' + str(port) in l]
        self.deviceLines = len(lines)
        self.timeout = 10.0
        
        devStr = str(device) + '/port' + str(port) + '/line0:' + str(self.deviceLines-1)
        self.CreateDIChan(devStr,'',DAQmx_Val_ChanForAllLines)
        
        self.AutoRegisterDoneEvent(0)
        
    def Read(self):
        # reads the current setting of all inputs
        data = np.zeros(self.deviceLines, dtype=np.uint8)
        bytesPerSample = c_long()
        samplesPerChannel = c_long()
        self.ReadDigitalLines(1,self.timeout,DAQmx_Val_GroupByChannel,data,self.deviceLines,samplesPerChannel,bytesPerSample,None)
        return data
        
    def DoneCallback(self, status):
        print "Status",status.value
        return 0 # the function should return an integer


class DigitalOutputs(Task):
    '''
    create DigitalOutputs object for all lines on port

    example:
    import nidaq
    do = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='high')
    do.StartTask()
    do.Write(np.array([True,False,True,False]))
    do.WriteBit(0,True)
    do.StopTask()
    do.ClearTask()
    '''

    def __init__(self, 
                device='Dev1', 
                port=1,
                initialState='low'):

        # construct task
        Task.__init__(self)
        
        # set up task properties
        self.port = port
        lines = GetDOLines(device)
        lines = [l for l in lines if 'port' + str(port) in l]
        self.deviceLines = len(lines)  
        self.timeout = 10.0
        
        #create initial state of output lines
        if initialState.lower()=='low':
            self.lastOut = np.zeros(self.deviceLines,dtype=np.uint8)
        elif initialState.lower()=='high':
            self.lastOut = np.ones(self.deviceLines,dtype=np.uint8)
        elif type(initial_state) == np.ndarray:
            self.lastOut = initialState
        else:
            raise TypeError("Initial state not understood. Try 'high' or 'low'")

        devStr = str(device) + "/port" + str(port) + "/line0:" + str(self.deviceLines-1)
        self.CreateDOChan(devStr,"",DAQmx_Val_ChanForAllLines)
        
        self.AutoRegisterDoneEvent(0)

    def DoneCallback(self,status):
        print "Status", status.value
        return 0
    
    def Write(self,levels):
        '''Writes a numpy uint8 array of levels to set the current output state'''
        self.lastOut = levels
        self.WriteDigitalLines(1,1,self.timeout,DAQmx_Val_GroupByChannel,levels,None,None)

    def WriteBit(self,index,level):
        '''Writes a single bit to the given line index'''
        self.lastOut[index] = level
        self.WriteDigitalLines(1,1,self.timeout,DAQmx_Val_GroupByChannel,self.lastOut,None,None)


if __name__ == '__main__':
    pass