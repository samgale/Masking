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
        assert(rigName in ('pilot','box5','sam'))
        self.rigName = rigName
        self.subjectName = None
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.drawDiodeBox = True
        self.monBackgroundColor = 0
        self.wheelRadius = 30.0 # mm
        self.minWheelAngleChange = 0 # radians per frame
        self.maxWheelAngleChange = 0.5 # radians per frame
        self.rewardsEnabled = True
        self.spacebarRewardsEnabled = True
        self.solenoidOpenTime = 0.05 # seconds
        if self.rigName=='pilot':
            self.saveDir = r'C:\Users\SVC_CCG\Desktop\Data' # path where parameters and data saved
            self.screen = 1 # monitor to present stimuli on
            self.monWidth = 53.34 # cm
            self.monDistance = 21.59 # cm
            self.monGamma = 2.3 # float or None
            self.monSizePix = (1920,1080)
            self.flipScreenHorz = False
            self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
            self.warpFile = None
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (935,-515)
            self.wheelPolarity = -1
            self.nidaqDevices = ('USB-6001','USB-6001')
            self.nidaqDeviceNames = ('Dev1','Dev2')
        elif self.rigName=='box5':
            self.saveDir = r'C:\Users\svc_ccg\Documents\Data'
            self.screen = 0 # monitor to present stimuli on
            self.monWidth = 50.8 # cm
            self.monDistance = 21.6 # cm
            self.monGamma = 2.1 # float or None
            self.monSizePix = (1920,1080)
            self.flipScreenHorz = False
            self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
            self.warpFile = None
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (935,-515)
            self.wheelPolarity = -1
            self.nidaqDevices = ('USB-6001',)
            self.nidaqDeviceNames = ('Dev1',)
        elif self.rigName=='sam':
            self.saveDir = r'C:\Users\svc_ccg\Desktop\Data'
            self.screen = 0 # monitor to present stimuli on
            self.monWidth = 53.34 # cm
            self.monDistance = 21.59 # cm
            self.monGamma = 2.1 # float or None
            self.monSizePix = (1920,1080)
            self.flipScreenHorz = False
            self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
            self.warpFile = None
            self.drawDiodeBox = False
            self.wheelPolarity = 1
            self.nidaqDevices = ('USB-6009',)
            self.nidaqDeviceNames = ('Dev1',)
            self.rewardsEnabled = False
    
    def prepareSession(self):
        self._win = None
        self._nidaqTasks = []
        
        startTime = time.localtime()
        self.startTime = time.strftime('%Y%m%d_%H%M%S',startTime)
        print('start time was: ' + time.strftime('%I:%M',startTime))
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / math.degrees(math.atan(0.5 * self.monWidth / self.monDistance))
        
        self.prepareWindow()
        
        if self.drawDiodeBox:
            self._diodeBox = visual.Rect(self._win,
                                         units='pix',
                                         width=self.diodeBoxSize,
                                         height=self.diodeBoxSize,
                                         lineColor=0,
                                         fillColor=1, 
                                         pos=self.diodeBoxPosition)
        
        self._keys = [] # list of keys pressed since previous frame
        self._mouse = event.Mouse(win=self._win)
                                    
        self.startNidaqDevice()
        self.rotaryEncoderRadians = []
        self.deltaWheelPos = [] # change in wheel position (angle translated to screen pixels)
        self.lickFrames = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.rewardFrames = [] # index of frames at which reward delivered
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        self.rewardSize = [] # size (solenoid open time) of each reward
        self._tone = False # tone triggered at next frame flip if True
        self._noise = False # noise triggered at next frame flip if True
        self._opto = False # False or dictionary of params for optoPulse at next frame flip
        
    
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
                                  units='pix',
                                  color=self.monBackgroundColor)
        self._warper = Warper(self._win,warp=self.warp,warpfile=self.warpFile)
        for _ in range(10):
            self._win.flip()
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
        
        # spacebar delivers reward
        # escape key ends session
        self._keys = event.getKeys()
        if self.spacebarRewardsEnabled and 'space' in self._keys and not self._reward:
            self._reward = self.solenoidOpenTime
            self.manualRewardFrames.append(self._sessionFrame)
        if 'escape' in self._keys:   
            self._continueSession = False
            
        if self._tone:
            self._toneOutput.write(True)
        elif self._noise:
            self._noiseOutput.write(True)
        
        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()
        
        self._frameSignalOutput.write(False)
        
        if self._opto:
            self.optoPulse(**self._opto)
            self._opto = False
        
        if self._reward and self.rewardsEnabled:
            self.triggerReward(self._reward)
            self.rewardFrames.append(self._sessionFrame)
            self.rewardSize.append(self._reward)
            self._reward = False
        
        if self._tone:
            self._toneOutput.write(False)
            self._tone = False
        elif self._noise:
            self._noiseOutput.write(False)
            self._noise = False
            
        self._sessionFrame += 1
        self._trialFrame += 1
                                               
    
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
        # Dev1 analog inputs
        # AI0: rotary encoder
        aiSampleRate = 2000 if self._win.monitorFramePeriod < 0.0125 else 1000
        aiBufferSize = 16
        self._rotaryEncoderInput = nidaqmx.Task()
        self._rotaryEncoderInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceNames[0]+'/ai0',min_val=0,max_val=5)
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
        
        # Dev1 analog outputs
        # AO0: water reward solenoid
        if self.nidaqDevices[0]=='USB-6001':
            aoSampleRate = 1000
            self._rewardOutput = nidaqmx.Task()
            self._rewardOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[0]+'/ao0',min_val=0,max_val=5)
            self._rewardOutput.write(0)
            self._rewardOutput.timing.cfg_samp_clk_timing(aoSampleRate)
            self._nidaqTasks.append(self._rewardOutput)
        
        # Dev2 analog outputs
        # AO0: led1
        # AO1: led2
        if len(self.nidaqDevices)>1 and self.nidaqDevices[1]=='USB-6001':
            self._optoOutput = nidaqmx.Task()
            self._optoOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[1]+'/ao0:1',min_val=0,max_val=5)
            self._optoOutput.write([0,0])
            self._optoAmp = 0
            self._optoOutput.timing.cfg_samp_clk_timing(aoSampleRate)
            self._nidaqTasks.append(self._optoOutput)
            
        # Dev1 digital inputs (port 0)
        # line 0.0: lick input
        self._lickInput = nidaqmx.Task()
        self._lickInput.di_channels.add_di_chan(self.nidaqDeviceNames[0]+'/port0/line0',
                                                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._nidaqTasks.append(self._lickInput)
        
        # Dev1 digital outputs (port 1)
        # line 1.0: frame signal
        self._frameSignalOutput = nidaqmx.Task()
        self._frameSignalOutput.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line0',
                                                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._frameSignalOutput.write(False)
        self._nidaqTasks.append(self._frameSignalOutput)
        
        # line 1.1: tone trigger
        self._toneOutput = nidaqmx.Task()
        self._toneOutput.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line1',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._toneOutput.write(False)
        self._nidaqTasks.append(self._toneOutput)
        
        # line 1.2: noise trigger
        self._noiseOutput = nidaqmx.Task()
        self._noiseOutput.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line2',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._noiseOutput.write(False)
        self._nidaqTasks.append(self._noiseOutput)
    
    
    def stopNidaqDevice(self):
        if getattr(self,'_optoAmp',0):
            self.optoOff()
        for task in self._nidaqTasks:
            task.close()
            
            
    def openSolenoid(self):
        self._solenoid = nidaqmx.Task()
        self._solenoid.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[0]+'/ao0',min_val=0,max_val=5)
        self._solenoid.write(5)
        
    
    def closeSolenoid(self):
        if getattr(self,'_solenoid',0):
            self._solenoid.write(0)
            self._solenoid.stop()
            self._solenoid.close()
            self._solenoid = None
            
    
    def waterTest(self,numPulses=100,pulseInterval=240):
        try:
            self.prepareSession()
            while self._continueSession:
                if self._sessionFrame > 0 and not self._sessionFrame % pulseInterval:
                    if len(self.rewardFrames) < numPulses:
                        self._reward = self.solenoidOpenTime
                    else:
                        self._continueSession = False
                self.showFrame()    
        except:
            raise
        finally:
            self.completeSession()
        
        
    def triggerReward(self,openTime):
        sampleRate = self._rewardOutput.timing.samp_clk_rate
        nSamples = int(openTime * sampleRate) + 1
        s = np.zeros(nSamples)
        s[:-1] = 5
        self._rewardOutput.stop()
        self._rewardOutput.timing.samp_quant_samp_per_chan = nSamples
        self._rewardOutput.write(s,auto_start=True)
        
        
    def optoOn(self,ch=[0,1],amp=5,ramp=0):
        self.optoPulse(ch,amp,onRamp=ramp,lastVal=amp)
    
    
    def optoOff(self,ch=[0,1],ramp=0):
        amp = self._optoAmp if ramp > 0 else 0 
        self.optoPulse(ch,amp,offRamp=ramp)
    
    
    def optoPulse(self,ch=[0,1],amp=5,dur=0,onRamp=0,offRamp=0,lastVal=0):
        sampleRate = self._optoOutput.timing.samp_clk_rate
        nSamples = int((dur + onRamp + offRamp) * sampleRate) + 1
        if nSamples < 2:
            nSamples = 2
        pulse = np.zeros((2,nSamples))
        pulse[ch,:-1] = amp
        pulse[ch,-1] = lastVal
        if onRamp > 0:
            ramp = np.linspace(0,amp,int(onRamp * sampleRate))
            pulse[ch,:ramp.size] = ramp
        if offRamp > 0:
            ramp = np.linspace(amp,0,int(offRamp * sampleRate))
            pulse[ch,-(ramp.size+1):-1] = ramp
        self._optoOutput.stop()
        self._optoOutput.timing.samp_quant_samp_per_chan = nSamples
        self._optoOutput.write(pulse,auto_start=True)
        self._optoAmp = lastVal
    
        
    def getNidaqData(self):
        # analog
        if self._rotaryEncoderData is None:
            encoderAngle = np.nan
        else:
            encoderData = np.array(self._rotaryEncoderData)
            encoderData *= 2 * math.pi / 5
            encoderAngle = np.arctan2(np.mean(np.sin(encoderData)),np.mean(np.cos(encoderData)))
        self.rotaryEncoderRadians.append(encoderAngle)
        self.deltaWheelPos.append(self.calculateWheelChange())
        
        # digital
        if self._lickInput.read():
            self.lickFrames.append(self._sessionFrame)
        
    
    def calculateWheelChange(self):
        # calculate angular change in wheel position
        if len(self.rotaryEncoderRadians) < 2 or np.isnan(self.rotaryEncoderRadians[-1]):
            angleChange = 0
        else:
            angleChange = self.rotaryEncoderRadians[-1] - self.rotaryEncoderRadians[-2]
            if angleChange < -math.pi:
                angleChange += 2 * math.pi
            elif angleChange > math.pi:
                angleChange -= 2 * math.pi
            if self.minWheelAngleChange < abs(angleChange) < self.maxWheelAngleChange:
                angleChange *= self.wheelPolarity
            else:
                angleChange = 0
        return angleChange
        


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
                if val is None:
                    val = np.nan
                try:
                    if isinstance(val,(list,tuple)) and all(isinstance(v,str) for v in val):
                        fileOut.create_dataset(paramName,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=str))
                    else:
                        try:
                            fileOut.create_dataset(paramName,data=val)
                        except:
                            fileOut.create_dataset(paramName,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=float))
                except:
                    print('\n' + 'could not save ' + key)
                    

if __name__ == "__main__":
    pass