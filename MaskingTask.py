# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools, math, random
import numpy as np
from psychopy import visual
from TaskControl import TaskControl


class MaskingTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        
        # parameters that can vary across trials are lists
        # only one of targetPos and targetOri can be len() > 1        
        
        self.preStimFrames = 240 # time between end of previous trial and stimulus onset
        self.maxResponseWaitFrames = 3600 # max time between stimulus onset and end of trial
        self.openLoopFrames = 30 # number of frames after stimulus onset before wheel movement has effects
        self.normRewardDistance = 0.25 # normalized to screen width
        
        # mouse can move target stimulus with wheel for early training
        # varying stimulus duration and/or masking not part of this stage
        self.moveStim = False
        self.keepTargetOnScreen = True
        self.postRewardTargetFrames = 1 # frames to freeze target after reward
        
        # target stimulus params
        self.normTargetPos = [(0,0)] # normalized initial xy position of target; center (0,0), bottom-left (-0.5,-0.5), top-right (0.5,0.5)
        self.targetFrames = [2] # duration of target stimulus; ignored if moveStim is True
        self.targetContrast = [1]
        self.targetSize = 100 # degrees
        self.targetSF = 0.04 # cycles/deg
        self.targetOri = [-45,45] # clockwise degrees from vertical
        
        # mask params
        self.maskType = 'plaid' # None, 'plaid', or 'noise'
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskOnset = [np.nan,0,2,4,8,16] # frames >=0 relative to target stimulus onset or NaN for no mask
        self.maskFrames = 9 # duration of mask  
        self.maskContrast = 1

    
    def setDefaultParams(self,taskVersion):
        if taskVersion == 'training1':
            self.setDefaultParams('pos')
            self.moveStim = True
            self.normRewardDistance = 0.15
            self.postRewardTargetFrames = 60
        elif taskVersion in ('pos','position'):
            self.targetOri = [0]
            self.normTargetPos = [(-0.25,0),(0.25,0)]
            self.normRewardDistance = 0.25
        elif taskVersion in ('ori','orientatation'):
            self.targetOri = [-45,45]
            self.normTargetPos = [(0,0)]
            self.normRewardDistance = 0.25
        else:
            print(str(taskVersion)+' is not a recognized version of this task')
    
    
    def checkParamValues(self):
        assert((len(self.normTargetPos)>1 and len(self.targetOri)==1) or
               (len(self.normTargetPos)==1 and len(self.targetOri)>1))
        

    def taskFlow(self):
        self.checkParamValues()        
        
        # create target stimulus
        targetPosPix = [tuple(p[i] * self.monSizePix[i] for i in (0,1)) for p in self.normTargetPos]
        targetSizePix = int(self.targetSize * self.pixelsPerDeg)
        sf = self.targetSF / self.pixelsPerDeg
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask='gauss',
                                    tex='sin',
                                    size=targetSizePix, 
                                    sf=sf)  
        
        # create mask
        # 'target' mask overlaps target
        # 'surround' mask surrounds but does not overlap target
        # 'full' mask surrounds and overlaps target
        if self.maskShape=='target':
            maskPos = targetPosPix
            maskSize = targetSizePix
            maskEdgeBlur = 'gauss'
        else:
            maskPos = [(0,0)]
            maskSize = max(self.monSizePix)
            maskEdgeBlur = 'none'
        
        if self.maskType=='noise':
            maskSize = 2**math.ceil(math.log(maskSize,2))
        
        if self.maskType=='plaid':
            maskOri = (0,90) if len(self.normTargetPos)>1 else (-45,45)
            mask = [visual.GratingStim(win=self._win,
                                       units='pix',
                                       mask=maskEdgeBlur,
                                       tex='sin',
                                       size=maskSize,
                                       contrast=self.maskContrast,
                                       sf=sf,
                                       ori=ori,
                                       opacity=opa,
                                       pos=pos) 
                                       for pos in maskPos
                                       for ori,opa in zip(maskOri,(1.0,0.5))]
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
                                     contrast=self.maskContrast)
                                     for pos in maskPos]
        
        if self.maskShape=='surround':
            mask += [visual.Circle(win=self._win,
                                   units='pix',
                                   radius=0.5*targetSizePix,
                                   lineColor=0.5,
                                   fillColor=0.5)
                                   for pos in targetPosPix]
        
        # create list of trial parameter combinations
        trialParams = list(itertools.product(targetPosPix,
                                             self.targetContrast,
                                             self.targetOri,
                                             self.targetFrames,
                                             self.maskOnset))
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialTargetPos = []
        self.trialTargetContrast = []
        self.trialTargetOri = []
        self.trialTargetFrames = []
        self.trialMaskOnset = []
        self.trialRewardDir = []
        self.trialRewardDist = []
        self.trialResponse = []
        self.rewardFrames = [] # index of frames at which reward earned
        
        monHalfWidth = 0.5 * self.monSizePix[0]
        
        while self._continueSession: # each loop is a frame presented on the monitor
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                trialIndex = len(self.trialStartFrame) % len(trialParams)
                if trialIndex == 0:
                    random.shuffle(trialParams)
                initTargetPos,targetContrast,targetOri,targetFrames,maskOnset = trialParams[trialIndex]
                targetPos = list(initTargetPos)
                closedLoopWheelPos = 0 # movment of wheel (translated to pixels) relative to initial target postion                
                if len(self.normTargetPos) > 1:
                    rewardDir = -1 if targetPos[0] > 0 else 1
                else:
                    rewardDir = -1 if targetOri < 0 else 1
                rewardDist = self.normRewardDistance * self.monSizePix[0]
                target.pos = targetPos
                target.contrast = targetContrast
                target.ori = targetOri
                if self.maskType == 'noise':
                    for m in mask:
                        if isinstance(m,visual.NoiseStim):
                            m.updateNoise()
                self.trialStartFrame.append(self._sessionFrame)
                self.trialTargetPos.append(targetPos)
                self.trialTargetContrast.append(targetContrast)
                self.trialTargetOri.append(targetOri)
                self.trialTargetFrames.append(targetFrames)
                self.trialMaskOnset.append(maskOnset)
                self.trialRewardDir.append(rewardDir)
                self.trialRewardDist.append(rewardDist)
                hasResponded = False
            
            # if gray screen period is complete, update target and mask stimuli
            if not hasResponded and self._trialFrame >= self.preStimFrames:
                if self._trialFrame >= self.preStimFrames + self.openLoopFrames:
                    closedLoopWheelPos += self.deltaWheelPos[-1]
                    if self.moveStim:
                        targetPos[0] += self.deltaWheelPos[-1]
                        if self.keepTargetOnScreen and abs(targetPos[0]) > monHalfWidth:
                            edge = monHalfWidth if targetPos[0] > 0 else -monHalfWidth
                            adjust = targetPos[0] - edge
                            targetPos[0] -= adjust
                            closedLoopWheelPos -= adjust
                        target.pos = targetPos
                if self.moveStim:
                    target.draw()
                else:
                    if (self.maskType is not None and not np.isnan(maskOnset) and 
                        (self.preStimFrames + maskOnset <= self._trialFrame < 
                         self.preStimFrames + maskOnset + self.maskFrames)):
                        for m in mask:
                            m.draw()
                    elif self._trialFrame < self.preStimFrames + targetFrames:
                        target.draw()
            
                # define response if wheel moved past threshold (either side) or max trial duration reached          
                if abs(closedLoopWheelPos) > rewardDist:
                    if closedLoopWheelPos * rewardDir > 0:
                        self.trialResponse.append(1) # correct
                        self.rewardFrames.append(self._sessionFrame)
                        self._reward = True
                        hasResponded = True
                    elif not self.keepTargetOnScreen:
                        self.trialResponse.append(-1) # incorrect
                        hasResponded = True
                elif self._trialFrame == self.preStimFrames + self.maxResponseWaitFrames:
                    self.trialResponse.append(0) # no response
                    hasResponded = True
                
            # show any post response stimuli
            if hasResponded:
                if (self.trialResponse[-1] > 0 and
                    self._sessionFrame < self.rewardFrames[-1] + self.postRewardTargetFrames):
                    targetPos[0] = initTargetPos[0] + rewardDist * rewardDir
                    target.pos = targetPos
                    target.draw()
                else:
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
            
            self.showFrame()


if __name__ == "__main__":
    pass