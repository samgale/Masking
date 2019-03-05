# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import math, os, random, time
import h5py
import numpy as np
from psychopy import monitors, visual
import ProjectorWindow
import nidaq
from threading import Timer


class TaskControl():
    
    def __init__(self):
        self.rig = 'pilot' # 'pilot' or 'np3'
        self._save = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.frameRate = 60.0
        self._screen = 1 # monitor to present stimuli on
        self.drawDiodeBox = True
        self.nidaqDevice = 'USB-6009'
        self.wheelRotDir = 1 # 1 or -1
        self.wheelSpeedGain = 20 # arbitrary scale factor
        self.rewardDur = 0.01 # duration in seconds of analog output pulse controlling reward size
        if self.rig=='pilot':
            self._saveDir = 'C:\Users\SVC_CCG\Desktop\Data' # path where parameters and data saved
            self.monWidth = 38.1 # cm
            self.monDistance = 15.24 # cm
            self.monGamma = None # float or None
            self.monSizePix = (1680,1050)
            self._flipScreenHorz = False
            self.warp = 'Disabled' # one of ('Disabled','Spherical','Cylindrical','Curvilinear','Warpfile')
            self.warpFile = None
            self.diodeBoxSize = 30
            self.diodeBoxPosition = (-600,-500)
            self.pixelsPerDeg = 1050/25
            
    def visStimFlip(self):
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.lineColor = self._diodeBox.fillColor
            self._diodeBox.draw()
        self.setAcquisitionSignal(1)
        self._win.flip()
        self.setAcquisitionSignal(0)
        
    def prepareRun(self):
        self.startTime = time.strftime('%Y%m%d_%H%M%S')
        
        self.prepareWindow()

        self._diodeBox = visual.Rect(self._win,
                                    units='pix',
                                    width=self.diodeBoxSize,
                                    height=self.diodeBoxSize,
                                    lineColor=0,
                                    fillColor=1, 
                                    pos=self.diodeBoxPosition)
        
        self.numpyRandomSeed = random.randint(0,2**32)
        self._numpyRandom = np.random.RandomState(self.numpyRandomSeed)
        
        self.startNidaqDevice()
        self.rotaryEncoderRadians = []
        
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',
                                     width=self.monWidth,
                                     distance=self.monDistance,
                                     gamma=self.monGamma)
        self._mon.setSizePix(self.monSizePix)
        self._mon.saveMon()
        self._win =  ProjectorWindow.ProjectorWindow(monitor=self._mon,
                                                     screen=self._screen,
                                                     fullscr=True,
                                                     flipHorizontal=self._flipScreenHorz,
                                                     warp=getattr(ProjectorWindow.Warp,self.warp),
                                                     warpfile=self.warpFile,
                                                     units='pix')      
        self._win.setRecordFrameIntervals(self.saveFrameIntervals)
                                               
    def completeRun(self):
        self._win.close()
        self.stopNidaqDevice()
        if self._save:
            fileOut = h5py.File(os.path.join(self._saveDir,'VisStim_'+self.__class__.__name__+'_'+self.startTime+'.hdf5'),'w')
            saveParameters(fileOut,self.__dict__)
            if self.saveFrameIntervals:
                fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
            fileOut.close()
        
    def startNidaqDevice(self):
        # rotary encoder: AI0
        self._rotEncoderInput = nidaq.AnalogInput(device='Dev1',channel=0,voltageRange=(0,5),sampRate=1000.0,bufferSize=8)
        self._rotEncoderInput.StartTask()
        
        # digital inputs (port 0)
        # line 0: lick detector
        self._digInputs = nidaq.DigitalInputs(device='Dev1',port=0)
        self._digInputs.StartTask()
        
        # digital outputs (port 1)
        # line 0: acquisition signal
        # line 1: reward solenoid
        self._digOutputs = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='low')
        self._digOutputs.StartTask()
        self._digOutputs.Write(self._digOutputs.lastOut)
    
    def stopNidaqDevice(self):
        for task in ['_rotEncoderInput','_digInputs','_digOutputs']:
            getattr(self,task).StopTask()
            getattr(self,task).ClearTask()
        
    def readRotaryEncoder(self):
        return self._rotEncoderInput.data[:]
        
    def saveEncoderAngle(self):
        encoderAngle = self.readRotaryEncoder()*2*math.pi/5
        self.rotaryEncoderRadians.append(np.arctan2(np.mean(np.sin(encoderAngle)),np.mean(np.cos(encoderAngle))))
        
    def translateEndoderChange(self):
        # translate encoder angle change to number of pixels to move visual stimulus
        if len(self.rotaryEncoderRadians)<2:
            pixelsToMove = 0
        else:
            angleChange = self.rotaryEncoderRadians[-1]-self.rotaryEncoderRadians[-2]
            if angleChange<-np.pi:
                angleChange += 2*math.pi
            elif angleChange>np.pi:
                angleChange -= 2*math.pi
            pixelsToMove = angleChange*self.wheelRotDir*self.wheelSpeedGain
        return pixelsToMove

    def deliverReward(self):
        rewardTime = 0.1
        self._digOutputs.WriteBit(1,1)
        t = Timer(rewardTime,self.endReward)
        t.start()

    def endReward(self):
        self._digOutputs.WriteBit(1,0)
        
    def getLickInput(self):
        return self._digInputs.Read()[0]
        
    def setAcquisitionSignal(self,level):
        self._digOutputs.WriteBit(0,level)
        

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