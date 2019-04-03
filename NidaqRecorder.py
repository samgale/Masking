# -*- coding: utf-8 -*-
"""
# record data
import NidaqRecorder
r = NidaqRecorder.NidaqRecorder()
r.start()
r.stop()

# read data
dataFile,analogDataset,sampleRate,channelNames = NidaqRecorder.readData()
ch0 = analogDataset[:,0]
# do stuff, then
dataFile.close()
"""

import fileIO
import h5py, time
from functools import partial
import nidaq
import numpy as np


def readData():
    filePath = fileIO.getFile()
    dataFile = h5py.File(filePath,'r')
    analogDataset = dataFile['AnalogInput']
    sampleRate = analogDataset.attrs.get('sampleRate')
    channelNames = analogDataset.attrs.get('channelNames')
    return dataFile,analogDataset,sampleRate,channelNames


def saveData(dataset,data):
    dataset.resize(dataset.shape[0]+data.shape[0],axis=0)
    dataset[-data.shape[0]:] = data


class NidaqRecorder():
    
    def __init__(self):
        self.appendStartTime = True
        
        self.analogInputChannels = [0,1,2,3,4,5,6]
        self.analogInputNames = ('vsync',
                                 'photodiode',
                                 'rotaryEncoder',
                                 'cam1Saving',
                                 'cam2Saving',
                                 'cam1Exposure',
                                 'cam2Exposure')
        self.analogInputSampleRate = 2000.0
        self.analogInputBufferSize = 500
        self.analogInputRange = [-10.0,10.0]
        
    def start(self):
        dataFilePath = fileIO.saveFile(fileType='*.hdf5')
        startTime = time.strftime('%Y%m%d_%H%M%S')
        if self.appendStartTime:
            dataFilePath = dataFilePath[:-5]+'_'+startTime+'.hdf5'
        self.dataFile = h5py.File(dataFilePath,'w',libver='latest')
        self.dataFile.attrs.create('startTime',startTime)
        
        numChannels = len(self.analogInputChannels)
        analogDataset = self.dataFile.create_dataset('AnalogInput',
                                                     (0,numChannels),
                                                     maxshape=(None,numChannels),
                                                     dtype=np.float64,
                                                     chunks=(self.analogInputBufferSize,numChannels),
                                                     compression='gzip',
                                                     compression_opts=1)
        analogDataset.attrs.create('sampleRate',self.analogInputSampleRate)
        analogDataset.attrs.create('channelNames',self.analogInputNames)
        
        self.analogInput = nidaq.AnalogInput(device='Dev1',
                                             channels=self.analogInputChannels,
                                             clock_speed=self.analogInputSampleRate,
                                             buffer_size=self.analogInputBufferSize,
                                             voltage_range=self.analogInputRange,
                                             custom_callback=partial(saveData,analogDataset))
        self.analogInput.start()
        
    def stop(self):
        self.analogInput.clear()
        self.dataFile.close()


if __name__ == '__main__':
    pass