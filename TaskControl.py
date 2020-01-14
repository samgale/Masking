# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import math, os, time
import h5py
import numpy as np
from psychopy import monitors, visual, event
from psychopy.visual.windowwarp import Warper
import nidaqmx


class TaskControl():
    
    def __init__(self,rigName):
        assert(rigName in ('pilot','box5'))
        self.rigName = rigName
        self.subjectName = None
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.drawDiodeBox = True
        self.nidaqDevice = 'USB-6001'
        self.nidaqDeviceName = 'Dev1'
        self.wheelRotDir = -1 # 1 or -1
        self.wheelSpeedGain = 1200 # arbitrary scale factor
        self.minWheelAngleChange = 0 # radians
        self.maxWheelAngleChange = 0.5 # radians
        self.spacebarRewardsEnabled = True
        self.solenoidOpenTime = 0.05 # seconds
        self._solenoid = None
        self.useLED = False
        self.ledDur = 1.0
        self.ledRamp = 0.1
        self.ledAmp = 5.0
        if self.rigName=='pilot':
            self.saveDir = r'C:\Users\SVC_CCG\Desktop\Data' # path where parameters and data saved
            self.screen = 1 # monitor to present stimuli on
            self.monWidth = 47.2 # cm
            self.monDistance = 21.0 # cm
            self.monGamma = None # float or None
            self.monSizePix = (1680,1050)
            self.flipScreenHorz = False
            self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
            self.warpFile = None
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (815,-500)
        elif self.rigName=='box5':
            self.saveDir = r'C:\Users\svc_ccg\Documents\Data'
            self.screen = 0 # monitor to present stimuli on
            self.monWidth = 50.8 # cm
            self.monDistance = 21.6 # cm
            self.monGamma = None # float or None
            self.monSizePix = (1920,1080)
            self.flipScreenHorz = False
            self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
            self.warpFile = None
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (931,-514)
        
    
    def prepareSession(self):
        self._win = None
        self._nidaqTasks = []
        
        startTime = time.localtime()
        self.startTime = time.strftime('%Y%m%d_%H%M%S',startTime)
        print('start time was: ' + time.strftime('%I:%M',startTime))
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / math.degrees(math.atan(0.5 * self.monWidth / self.monDistance))
        
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
        self.lickFrames = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        self._led = False # led triggered at next frame flip if True
        self._tone = False # tone triggered at next frame flip if True
        self._noise = False # noise triggered at next frame flip if True
        
    
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',
                                     width=self.monWidth,
                                     distance=self.monDistance,
                                     gamma=self.monGamma)
        self._mon.setSizePix(self.monSizePix)
        self._mon.saveMon()
        self._win = visual.Window(monitor=self._mon,
                                  screen=self.screen,
                                  fullscr=True,
                                  flipHorizontal=self.flipScreenHorz,
                                  units='pix')
        self._warper = Warper(self._win,warp=self.warp,warpfile=self.warpFile)
        for _ in range(10):
            self.frameRate = self._win.getActualFrameRate() # do this before recording frame intervals
            if self.frameRate is not None:
                break
        assert(self.frameRate is not None)
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
        self._frameSignalOutput.write(True)
        if self._tone:
            self._toneOutput.write(True)
        elif self._noise:
            self._noiseOutput.write(True)
        
        # spacebar delivers reward
        # escape key ends session
        keys = event.getKeys()
        if self.spacebarRewardsEnabled and 'space' in keys:
            self._reward = True
            self.manualRewardFrames.append(self._sessionFrame)
        if 'escape' in keys:   
            self._continueSession = False
        
        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()
        
        if self._reward:
            self.triggerReward()
            self._reward = False
        if self._led:
            self.triggerLED()
            self._led = False
        
        self._sessionFrame += 1
        self._trialFrame += 1
        
        self._frameSignalOutput.write(False)
        if self._tone:
            self._toneOutput.write(False)
            self._tone = False
        elif self._noise:
            self._noiseOutput.write(False)
            self._noise = False
                                               
    
    def completeSession(self):
        try:
            if self._win is not None:
                self._win.close()
            self.stopNidaqDevice()
        except:
            raise
        finally:
            if self.saveParams:
                subjName = '' if self.subjectName is None else self.subjectName + '_'
                filePath = os.path.join(self.saveDir,self.__class__.__name__ + '_' + subjName + self.startTime)
                fileOut = h5py.File(filePath+'.hdf5','w')
                saveParameters(fileOut,self.__dict__)
                if self.saveFrameIntervals and self._win is not None:
                    fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
                fileOut.close()
        
    
    def startNidaqDevice(self):
        # analog inputs
        # AI0: rotary encoder
        aiSampleRate = 2000 if self.frameRate > 100 else 1000
        aiBufferSize = 16
        self._rotaryEncoderInput = nidaqmx.Task()
        self._rotaryEncoderInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceName+'/ai0',min_val=0,max_val=5)
        self._rotaryEncoderInput.timing.cfg_samp_clk_timing(aiSampleRate,
                                                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                            samps_per_chan=aiBufferSize)
                                            
        def readRotaryEncoderBuffer(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
            self._rotaryEncoderData = self._rotaryEncoderInput.read(number_of_samples_per_channel=number_of_samples)
            return 0
        
        self._rotaryEncoderInput.register_every_n_samples_acquired_into_buffer_event(aiBufferSize,readRotaryEncoderBuffer)
        self._rotaryEncoderData = None
        self._rotaryEncoderInput.start()
        self._nidaqTasks.append(self._rotaryEncoderInput)
        
        # analog outputs
        # AO0: water reward solenoid trigger
        aoSampleRate = 1000
        rewardBufferSize = int(self.solenoidOpenTime * aoSampleRate) + 1
        self._rewardSignal = np.zeros(rewardBufferSize)
        self._rewardSignal[:-1] = 5
        self._rewardOutput = nidaqmx.Task()
        self._rewardOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceName+'/ao0',min_val=0,max_val=5)
        self._rewardOutput.write(0)
        self._rewardOutput.timing.cfg_samp_clk_timing(aoSampleRate,samps_per_chan=rewardBufferSize)
        self._nidaqTasks.append(self._rewardOutput)
        
        # AO1: led/laser trigger
        if self.useLED:
            ledBufferSize = int(self.ledDur * aoSampleRate) + 1
            ledRamp = np.linspace(0,self.ledAmp,int(self.ledRamp * aoSampleRate))
            self._ledSignal = np.zeros(ledBufferSize)
            self._ledSignal[:-1] = self.ledAmp
            if self.ledRamp > 0:
                self._ledSignal[:ledRamp.size] = ledRamp
                self._ledSignal[-(ledRamp.size+1):-1] = ledRamp[::-1]
            self._ledOutput = nidaqmx.Task()
            self._ledOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceName+'/ao1',min_val=0,max_val=5)
            self._ledOutput.write(0)
            self._ledOutput.timing.cfg_samp_clk_timing(aoSampleRate,samps_per_chan=ledBufferSize)
            self._nidaqTasks.append(self._ledOutput)
            
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
        self._nidaqTasks.append(self._frameSignalOutput)
        
        # line 1.1: tone trigger
        self._toneOutput = nidaqmx.Task()
        self._toneOutput.do_channels.add_do_chan(self.nidaqDeviceName+'/port1/line1',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._toneOutput.write(False)
        self._nidaqTasks.append(self._toneOutput)
        
        # line 1.2: noise trigger
        self._noiseOutput = nidaqmx.Task()
        self._noiseOutput.do_channels.add_do_chan(self.nidaqDeviceName+'/port1/line2',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._noiseOutput.write(False)
        self._nidaqTasks.append(self._noiseOutput)
    
    
    def stopNidaqDevice(self):
        for task in self._nidaqTasks:
            task.close()
            
            
    def openSolenoid(self):
        self._solenoid = nidaqmx.Task()
        self._solenoid.ao_channels.add_ao_voltage_chan(self.nidaqDeviceName+'/ao0',min_val=0,max_val=5)
        self._solenoid.write(5)
        
    
    def closeSolenoid(self):
        if self._solenoid is not None:
            self._solenoid.write(0)
            self._solenoid.close()
            self._solenoid = None
        
        
    def triggerReward(self):
        self._rewardOutput.stop()
        self._rewardOutput.write(self._rewardSignal,auto_start=True)
        
    
    def triggerLED(self):
        self._ledOutput.stop()
        self._ledOutput.write(self._ledSignal,auto_start=True)
    
        
    def getNidaqData(self):
        # analog
        if self._rotaryEncoderData is None:
            encoderAngle = np.nan
        else:
            encoderData = np.array(self._rotaryEncoderData)
            encoderData *= 2 * math.pi / 5
            encoderAngle = np.arctan2(np.mean(np.sin(encoderData)),np.mean(np.cos(encoderData)))
        self.rotaryEncoderRadians.append(encoderAngle)
        self.deltaWheelPos.append(self.translateEncoderChange())
        
        # digital
        if self._lickInput.read():
            self.lickFrames.append(self._sessionFrame)
        
    
    def translateEncoderChange(self):
        # translate encoder angle change to number of pixels to move visual stimulus
        if len(self.rotaryEncoderRadians) < 2 or np.isnan(self.rotaryEncoderRadians[-1]):
            pixelsToMove = 0
        else:
            angleChange = self.rotaryEncoderRadians[-1] - self.rotaryEncoderRadians[-2]
            if angleChange < -math.pi:
                angleChange += 2 * math.pi
            elif angleChange > math.pi:
                angleChange -= 2 * math.pi
            if self.minWheelAngleChange < abs(angleChange) < self.maxWheelAngleChange:
                pixelsToMove = angleChange * self.wheelRotDir * self.wheelSpeedGain
            else:
                pixelsToMove = 0
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
                    elif (isinstance(val,(list,tuple)) and len(val) > 1 and 
                          all(isinstance(v,(list,tuple)) for v in val) and [len(v) for v in val].count(len(val[0])) < len(val)):
                        # convert list of lists of unequal len to nan padded array
                        valArray = np.full((len(val),max(len(v) for v in val)),np.nan)
                        for i,v in enumerate(val):
                            valArray[i,:len(v)] = v
                        val = valArray
                    fileOut.create_dataset(paramName,data=val)
                except:
                    print('could not save ' + key)
                    

if __name__ == "__main__":
    pass