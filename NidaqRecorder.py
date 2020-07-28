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

import h5py, os, time
import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader


def readData(filePath):
    dataFile = h5py.File(filePath,'r')
    analogDataset = dataFile['AnalogInput']
    sampleRate = analogDataset.attrs.get('sampleRate')
    channelNames = analogDataset.attrs.get('channelNames')
    return dataFile,analogDataset,sampleRate,channelNames    


class NidaqRecorder():
    
    def __init__(self):
        self.saveDirPath = 'C:\\Users\svc_ccg\\Desktop\\Data'
        
        self.nidaqDevice = 'USB-6009'
        self.nidaqDeviceName = 'Dev1'
        
        self.analogInputNames = ('vsync',
                                 'photodiode',
                                 'rotaryEncoder',
                                 'cam1Saving',
                                 'cam2Saving',
                                 'cam1Exposure',
                                 'cam2Exposure',
                                 'led')
        self.analogInputSampleRate = 2000.0
        self.analogInputBufferSize = 500
        self.analogInputMin = -10.0
        self.analogInputMax = 10.0
        
    def start(self,fileName=None):
        startTime = time.strftime('%Y%m%d_%H%M%S')
        fileName = startTime if fileName is None else fileName+'_'+startTime
        dataFilePath = os.path.join(self.saveDirPath,fileName+'.hdf5')
        self.dataFile = h5py.File(dataFilePath,'w',libver='latest')
        self.dataFile.attrs.create('startTime',startTime)
        
        numChannels = len(self.analogInputNames)
        analogDataset = self.dataFile.create_dataset('AnalogInput',
                                                     (0,numChannels),
                                                     maxshape=(None,numChannels),
                                                     dtype=np.float64,
                                                     chunks=(self.analogInputBufferSize,numChannels),
                                                     compression='gzip',
                                                     compression_opts=1)
        analogDataset.attrs.create('sampleRate',self.analogInputSampleRate)
        analogDataset.attrs.create('channelNames',self.analogInputNames)
        
        self.analogInput = nidaqmx.Task()
        self.analogInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceName+'/ai0:'+str(numChannels-1),
                                                         terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                                         min_val=self.analogInputMin,
                                                         max_val=self.analogInputMax)
        self.analogInput.timing.cfg_samp_clk_timing(self.analogInputSampleRate,
                                                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                    samps_per_chan=self.analogInputBufferSize)
        
        analogInputReader = AnalogMultiChannelReader(self.analogInput.in_stream)
        analogInputData = np.zeros((numChannels,self.analogInputBufferSize))
                                            
        def saveAnalogData(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
            analogInputReader.read_many_sample(analogInputData,number_of_samples_per_channel=number_of_samples)
            analogDataset.resize(analogDataset.shape[0]+number_of_samples,axis=0)
            analogDataset[-number_of_samples:] = analogInputData.T
            return 0
        
        self.analogInput.register_every_n_samples_acquired_into_buffer_event(self.analogInputBufferSize,
                                                                             saveAnalogData)
        
        self.analogInput.start()

    
    def stop(self):
        self.analogInput.close()
        self.dataFile.close()


if __name__ == '__main__':
    pass