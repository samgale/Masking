# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools, math, random, traceback
from psychopy import visual, event
from TaskControl import TaskControl
import numpy as np


class MaskingTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        self.preStimFrames = 240 # time between end of previous trial and stimulus onset
        self.maxResponseWaitFrames = 360 # max time between stimulus onset and end of trial 
        self.rewardDistance = 5 # degrees to move stim for reward
        
        # mouse can move target stimulus with wheel for early training
        # varying stimulus duration and/or masking not part of this stage
        self.moveStim = False
        self.preMoveFrames = 30 # number of frames after stimulus onset before stimulus moves
        
        # stim params
        self.stimFrames = [2] # duration of target stimulus; ignored if moveStim is True
        self.stimContrast = [0.2]
        self.stimSize = 10 # degrees
        self.gratingsSF = 0.5 # cycles/deg
        self.gratingsOri = [-45,45] # clockwise degrees from vertical
        
        # mask params
        self.maskType = 'plaid' # None, 'plaid', or 'noise'
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskOnset = [np.nan,0,2,4,6,8,16] # frames >=0 relative to target stimulus onset, or NaN for no mask
        self.maskFrames = 9 # duration of mask      

    def checkParameterValues(self):
        pass
    
    def run(self,subjectName=None):
        if subjectName is not None:
            self.subjectName = subjectName
        
        self.checkParameterValues()
        
        self.prepareRun()
        
        try:
            # create stim
            stimSizePix = int(self.stimSize*self.pixelsPerDeg)
            sf = self.gratingsSF/self.pixelsPerDeg
            stim = visual.GratingStim(win=self._win,
                                      units='pix',
                                      mask='gauss',
                                      tex='sin',
                                      size=stimSizePix, 
                                      pos=(0,0),
                                      sf=sf)  
            
            # create mask
            if self.maskShape=='target':
                maskSize = stimSizePix
                maskEdgeBlur = 'gauss'
            else:
                maskSize = self.monSizePix[1]
                maskEdgeBlur = 'none'
            
            if self.maskType=='noise':
                maskSize = 2**math.ceil(math.log(maskSize,2))
            
            if self.maskType=='plaid':
                mask = [visual.GratingStim(win=self._win,
                                           units='pix',
                                           mask=maskEdgeBlur,
                                           tex='sin',
                                           size=maskSize,
                                           pos=(0,0),
                                           sf=sf,
                                           ori=ori,
                                           opacity=op) for ori,op in zip((-45,45),(1.0,0.5))]
            elif self.maskType=='noise':
                mask = [visual.NoiseStim(win=self._win,
                                          units='pix',
                                          mask=maskEdgeBlur,
                                          noiseType='Filtered',
                                          noiseFractalPower = 0,
                                          noiseFilterOrder = 1,
                                          noiseFilterLower = 0.5*sf,
                                          noiseFilterUpper = 2*sf,
                                          size=maskSize, 
                                          pos=(0,0))]
            
            if self.maskShape=='surround':
                mask.append(visual.Circle(win=self._win,
                                          units='pix',
                                          radius=stimSizePix/2,
                                          lineColor=0.5,
                                          fillColor=0.5,
                                          pos=(0,0)))
            
            # create list of trial parameter combinations
            trialParams = list(itertools.product(self.stimContrast,self.gratingsOri,self.stimFrames,self.maskOnset))
            
            # run session
            sessionFrame = 0
            trialFrame = 0
            self.trialStartFrame = []
            self.trialEndFrame = []
            self.trialStimContrast = []
            self.trialOri = []
            self.trialStimFrames = []
            self.trialMaskOnset = []
            self.trialRewardSide = []
            self.trialResponse = []
            
            while True: # each loop is a frame flip
                # get rotary encoder and digital input states
                self.getNidaqData()
                
                # start new trial
                if trialFrame == 0:
                    wheelPos = 0
                    trialIndex = len(self.trialStartFrame) % len(trialParams)
                    if trialIndex == 0:
                        random.shuffle(trialParams)
                    stimContrast,ori,stimFrames,maskOnset = trialParams[trialIndex]
                    rewardSide = -1 if ori < 0 else 1
                    stim.contrast = stimContrast
                    stim.ori = ori
                    stim.pos = (0,0)
                    if self.maskType == 'noise':
                        mask[0].updateNoise()
                    self.trialStartFrame.append(sessionFrame)
                    self.trialStimContrast.append(stimContrast)
                    self.trialOri.append(ori)
                    self.trialStimFrames.append(stimFrames)
                    self.trialMaskOnset.append(maskOnset)
                    self.trialRewardSide.append(rewardSide)
                
                # update stimulus/mask after pre-stimulus gray screen period is complete
                if trialFrame > self.preStimFrames:
                    if trialFrame > self.preStimFrames + self.preMoveFrames:
                        wheelPos += self.translateEndoderChange()
                    if self.moveStim:
                        stim.pos = (wheelPos,0)
                        stim.draw()
                    else:
                        if (self.maskType is not None and not np.isnan(maskOnset) and 
                           (self.preStimFrames + maskOnset < trialFrame <= 
                            self.preStimFrames + maskOnset + self.maskFrames)):
                            for m in mask:
                                m.draw()
                        elif trialFrame <= self.preStimFrames + stimFrames:
                            stim.draw()
                     
                self.visStimFlip()
                trialFrame += 1
                sessionFrame += 1
                
                # end trial if wheel moved past threshold (either side) or max trial duration reached
                if abs(wheelPos) > self.rewardDistance * self.pixelsPerDeg:
                    if wheelPos * rewardSide > 0:
                        self.triggerSound()
                        self.deliverReward()
                        self.trialResponse.append(1) # correct
                    else:
                        self.trialResponse.append(-1) # incorrect
                    self.trialEndFrame.append(trialFrame)
                    trialFrame = 0
                elif trialFrame == self.preStimFrames + self.maxResponseWaitFrames:
                    self.trialResponse.append(0) # no response
                    self.trialEndFrame.append(trialFrame)
                    trialFrame = 0
                
                # check for keyboard events to end session
                if len(event.getKeys()) > 0:                  
                    event.clearEvents()
                    break
                
        except:
            raise #traceback.print_exc()
            
        finally:
            self.completeRun()


if __name__ == "__main__":
    pass