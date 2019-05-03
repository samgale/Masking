# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import math, os, random, time
import h5py
import numpy as np
from psychopy import monitors, visual, event
import ProjectorWindow
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader


class TaskControl():
    
    def __init__(self):
        self.rig = 'pilot' # 'pilot' or 'np3'
        self.subjectName = None
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.screen = 1 # monitor to present stimuli on
        self.drawDiodeBox = True
        self.nidaqDevice = 'USB-6001'
        self.nidaqDeviceName = 'Dev1'
        self.wheelRotDir = -1 # 1 or -1
        self.wheelSpeedGain = 1200 # arbitrary scale factor
        self.maxWheelAngleChange = 0.5 # radians
        self.spacebarRewardsEnabled = True
        self.solenoidOpenTime = 0.05 # seconds
        if self.rig=='pilot':
            self.saveDir = 'C:\Users\SVC_CCG\Desktop\Data' # path where parameters and data saved
            self.monWidth = 47.2 # cm
            self.monDistance = 22.9 # cm
            self.monGamma = None # float or None
            self.monSizePix = (1680,1050)
            self.flipScreenHorz = False
            self.warp = 'Disabled' # one of ('Disabled','Spherical','Cylindrical','Curvilinear','Warpfile')
            self.warpFile = None
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (815,-500)
        elif self.rig=='np3':
            pass
        
    
    def prepareSession(self):
        self.startTime = time.strftime('%Y%m%d_%H%M%S')
        self._win = None
        self._nidaqTasks = []
        
        self.numpyRandomSeed = random.randint(0,2**32)
        self._numpyRandom = np.random.RandomState(self.numpyRandomSeed)
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / (math.tan(0.5 * self.monWidth / self.monDistance) * 180 / math.pi)
        
        self.prepareWindow()

        self._diodeBox = visual.Rect(self._win,
                                    units='pix',
                                    width=self.diodeBoxSize,
                                    height=self.diodeBoxSize,
                                    lineColor=0,
                                    fillColor=1, 
                                    pos=self.diodeBoxPosition)
                                    
        self.startNidaqDevice()
        self.rotaryEncoderRadians = []
        self.deltaWheelPos = [] # change in wheel position (angle translated to screen pixels)
        self.lickState = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        
        
    
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',
                                     width=self.monWidth,
                                     distance=self.monDistance,
                                     gamma=self.monGamma)
        self._mon.setSizePix(self.monSizePix)
        self._mon.saveMon()
        self._win =  ProjectorWindow.ProjectorWindow(monitor=self._mon,
                                                     screen=self.screen,
                                                     fullscr=True,
                                                     flipHorizontal=self.flipScreenHorz,
                                                     warp=getattr(ProjectorWindow.Warp,self.warp),
                                                     warpfile=self.warpFile,
                                                     units='pix')
        self.frameRate = self._win.getActualFrameRate() # do this before recording frame intervals
        self._win.setRecordFrameIntervals(self.saveFrameIntervals)
        
        
    def start(self,subjectName=None):
        try:
            if subjectName is not None:
                self.subjectName = str(subjectName)
            
            self.prepareSession()
            
            self.taskFlow()
        
        except:
            raise
            
        finally:
            self.completeSession()
    
    
    def taskFlow(self):
        # override this method in subclass
    
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # do stuff, for example:
            # check for licks and/or wheel movement
            # update/draw stimuli
            
            self.showFrame()
    
    
    def showFrame(self):
        # check for keyboard events
        # set frame acquisition and reward signals
        # flip frame buffer
        # update session and trial frame counters
        
        # spacebar delivers reward
        keys = event.getKeys()
        if self.spacebarRewardsEnabled and 'space' in keys:
            self._reward = True
            self.manualRewardFrames.append(self._sessionFrame)
        
        # set frame acquisition and reward signals 
        self._frameSignalOutput.write(True)
        if self._reward:
            self.deliverReward()
            self.reward = False
        
        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()
        
        # reset frame acquisition signal
        self._frameSignalOutput.write(False)
        
        self._sessionFrame += 1
        self._trialFrame += 1
        
        # escape key ends session
        if 'escape' in keys:   
            self._continueSession = False
                                               
    
    def completeSession(self):
        if self._win is not None:
            self._win.close()
        self.stopNidaqDevice()
        if self.saveParams:
            subjName = '' if self.subjectName is None else self.subjectName + '_'
            filePath = os.path.join(self.saveDir,self.__class__.__name__ + '_' + subjName + self.startTime)
            fileOut = h5py.File(filePath+'.hdf5','w')
            saveParameters(fileOut,self.__dict__)
            if self.saveFrameIntervals:
                fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
            fileOut.close()
        
    
    def startNidaqDevice(self):
        # analog inputs
        # AI0: rotary encoder
        aiSampleRate = 1000.0
        aiBufferSize = int(1 / self.frameRate * aiSampleRate)
        self._rotaryEncoderInput = nidaqmx.Task()
        self._rotaryEncoderInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceName+'/ai0',
                                                                 min_val=0,
                                                                 max_val=5)
        self._rotaryEncoderInput.timing.cfg_samp_clk_timing(aiSampleRate,
                                                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                            samps_per_chan=aiBufferSize)

        rotaryEncoderReader = AnalogSingleChannelReader(self._rotaryEncoderInput.in_stream)
        self._rotaryEncoderData = np.zeros(aiBufferSize)
                                            
        def readRotaryEncoderBuffer(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
            rotaryEncoderReader.read_many_sample(self._rotaryEncoderData,number_of_samples_per_channel=number_of_samples)
            return 0
        
        self._rotaryEncoderInput.register_every_n_samples_acquired_into_buffer_event(aiBufferSize,readRotaryEncoderBuffer)

        self._rotaryEncoderInput.start()
        self._nidaqTasks.append(self._rotaryEncoderInput)
        
        # analog outputs
        # AO0: water reward solenoid trigger
        aoSampleRate = 1000.0
        aoBufferSize = int(self.solenoidOpenTime * aoSampleRate) + 1
        self._rewardSignal = np.zeros(aoBufferSize)
        self._rewardSignal[:-1] = 5
        self._rewardOutput = nidaqmx.Task()
        self._rewardOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceName+'/ao0',min_val=0,max_val=5)
        self._rewardOutput.write(0)
        self._rewardOutput.timing.cfg_samp_clk_timing(1000,samps_per_chan=aoBufferSize)
        self._nidaqTasks.append(self._rewardOutput)
            
        # digital inputs (port 0)
        # line 0.0: lick input
        self._lickInput = nidaqmx.Task()
        self._lickInput.di_channels.add_di_chan(self.nidaqDeviceName+'/port0/line0',
                                                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._nidaqTasks.append(self._lickInput)
        
        # digital outputs (port 1)
        # line 1.0: frame signal
        self._frameSignalOutput = nidaqmx.Task()
        self._frameSignalOutput.do_channels.add_do_chan(self.nidaqDeviceName+'/port1/line0',
                                                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._frameSignalOutput.write(False)
        self._nidaqTasks.append(self._frameSignal)    
    
    
    def stopNidaqDevice(self):
        for task in self._nidaqTasks:
            task.close()
        
        
    def deliverReward(self):
        self._rewardOutput.stop()
        self._rewardOutput.write(self._rewardSignal,auto_start=True)
    
        
    def getNidaqData(self):
        # analog
        encoderAngle = self._rotaryEncoderData * 2 * math.pi / 5
        self.rotaryEncoderRadians.append(np.arctan2(np.mean(np.sin(encoderAngle)),np.mean(np.cos(encoderAngle))))
        self.deltaWheelPos.append(self.translateEndoderChange())
        
        # digital
        self.lickState.append(self._lickInput.read())
        
    
    def translateEndoderChange(self):
        # translate encoder angle change to number of pixels to move visual stimulus
        if len(self.rotaryEncoderRadians) < 2:
            pixelsToMove = 0
        else:
            angleChange = self.rotaryEncoderRadians[-1] - self.rotaryEncoderRadians[-2]
            if angleChange < -math.pi:
                angleChange += 2 * math.pi
            elif angleChange > math.pi:
                angleChange -= 2 * math.pi
            if angleChange > self.maxWheelAngleChange:
                pixelsToMove = 0
            else:
                pixelsToMove = angleChange * self.wheelRotDir * self.wheelSpeedGain
        return pixelsToMove
        


def saveParameters(fileOut,paramDict,dictName=None):
    for key,val in paramDict.items():
        if key[0] != '_':
            if dictName is None:
                paramName = key
            else:
                paramName = dictName+'_'+key
            if isinstance(val,dict):
                saveParameters(fileOut,val,paramName)
            else:
                try:
                    if val is None:
                        val = np.nan
                    fileOut.create_dataset(paramName,data=val)
                except:
                    print 'could not save ' + key
                    

if __name__ == "__main__":
    pass