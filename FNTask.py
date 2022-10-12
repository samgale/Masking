# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

import random
import numpy as np
from psychopy import visual    
from TaskControl import TaskControl


class FNTask(TaskControl):
    
    def __init__(self,rigName,taskVersion=None):
        TaskControl.__init__(self,rigName)
        
        self.taskVersion = taskVersion
        self.maxTrials = None
        self.spacebarRewardsEnabled = False
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 600 # max total preStim frames
        self.quiescentFrames = 60 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFrames = 18 # min frames after stimulus onset before wheel movement has effects
        self.maxResponseWaitFrames = 600 # max frames from stim onset to end of trial
        self.rewardDelayFrames = 0
        self.probCatch = 0.1 # probability of catch trials with normal trial timing but no stimulus or reward
        
        self.wheelRewardDistance = 8.0 # mm of wheel movement to achieve reward
        self.maxQuiescentMoveDist = 1.0 # max allowed mm of wheel movement during quiescent period
        
        self.targetStartPos = [-0.4,0] # normalized initial xy  position of target; center (0,0), bottom-left (-0.5,-0.5), top-right (0.5,0.5)
        self.targetRewardDistance = 0.8
        self.targetAutoMoveRate = 0 # fraction of normalized screen width per second that target automatically moves
        self.keepTargetOnScreen = True
        self.postRewardTargetFrames = 60 # frames to freeze target after reward
        self.targetContrast = 1
        self.targetSize = 20 # degrees
        self.targetSF = 0.1 # cycles/deg
        self.targetOri = 0
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)

    
    def setDefaultParams(self,taskVersion):
        if taskVersion == 'training1':
            # target moves automatically and each trial is autorewarded
            self.probCatch = 0
            self.quiescentFrames = 0
            self.maxResponseWaitFrames = 3600
            self.targetAutoMoveRate = 0.5
            
        elif taskVersion == 'training2':
            # offset target to one side and increase reward distance
            self.probCatch = 0
            self.quiescentFrames = 0
            self.maxResponseWaitFrames = 3600
            self.wheelRewardDistance = 8.0
            
        elif taskVersion == 'training3':
            # introduce quiescent period, catch trials, shorter response window, and longer wheel reward distance
            self.maxResponseWaitFrames = 600 # adjust this
            self.wheelRewardDistance = 16.0
            
        elif taskVersion == 'training4':
            self.wheelRewardDistance = 64.0
            
        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
    
     
    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create target stimulus
        targetStartPosPix = [p * s for p,s in zip(self.targetStartPos,self.monSizePix)]
        targetSizePix = self.targetSize * self.pixelsPerDeg
        edgeBlurWidth = {'fringeWidth': self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask=self.gratingEdge,
                                    maskParams=edgeBlurWidth,
                                    tex=self.gratingType,
                                    contrast=self.targetContrast,
                                    size=targetSizePix,
                                    sf=self.targetSF / self.pixelsPerDeg,
                                    ori=self.targetOri)  
        
        # calculate pixels to move or degrees to rotate stimulus per radian of wheel movement
        rewardDir = 1 if self.targetStartPos[0] < 0 else -1
        rewardMove = self.targetRewardDistance * self.monSizePix[0]
        self.wheelGain = rewardMove / (self.wheelRewardDistance / self.wheelRadius)
        maxQuiescentMove = (self.maxQuiescentMoveDist / self.wheelRadius) * self.wheelGain
        monitorEdge = 0.5 * (self.monSizePix[0] - targetSizePix)
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.isCatchTrial = []
        self.trialResponse = []
        self.trialResponseDir = []
        self.trialResponseFrame = []
        self.quiescentMoveFrames = [] # frames where quiescent period was violated
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # start a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames) # can grow larger than preStimFrames during quiescent period
                quiescentWheelMove = 0 # virtual (not on screen) change in target position during quiescent period
                closedLoopWheelMove = 0 # actual change in target position during closed loop period
                targetPos = targetStartPosPix[:]
                self.isCatchTrial.append(random.random() < self.probCatch)
                self.trialStartFrame.append(self._sessionFrame)
                hasResponded = False
            
            # extend pre stim gray frames if wheel moving during quiescent period
            if self.trialPreStimFrames[-1] - self.quiescentFrames < self._trialFrame < self.trialPreStimFrames[-1]:
                quiescentWheelMove += self.deltaWheelPos[-1] * self.wheelGain
                if abs(quiescentWheelMove) > maxQuiescentMove:
                    self.quiescentMoveFrames.append(self._sessionFrame)
                    self.trialPreStimFrames[-1] += randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                    quiescentWheelMove = 0
            
            if not hasResponded and self._trialFrame >= self.trialPreStimFrames[-1]:
                if self._trialFrame == self.trialPreStimFrames[-1]:
                    self.trialStimStartFrame.append(self._sessionFrame)
                
                if self._trialFrame >= self.trialPreStimFrames[-1] + self.openLoopFrames:
                    # update target stimulus position
                    if self.targetAutoMoveRate > 0:
                        deltaPos = rewardDir * self.targetAutoMoveRate * self.monSizePix[0] * self._win.monitorFramePeriod
                        targetPos[0] += deltaPos
                        closedLoopWheelMove += deltaPos
                    else:
                        deltaPos = self.deltaWheelPos[-1] * self.wheelGain
                        targetPos[0] += deltaPos
                        closedLoopWheelMove += deltaPos
                        if self.keepTargetOnScreen and abs(targetPos[0]) > monitorEdge:
                            adjust = targetPos[0] - monitorEdge if targetPos[0] > 0 else targetPos[0] + monitorEdge
                            targetPos[0] -= adjust
                            closedLoopWheelMove -= adjust
                            
                    # check for response past reward threshold
                    if self._trialFrame < self.trialPreStimFrames[-1] + self.maxResponseWaitFrames:
                        if abs(closedLoopWheelMove) > rewardMove:
                            moveDir = 1 if closedLoopWheelMove > 0 else -1
                            if moveDir == rewardDir or self.targetStartPos[0] == 0:
                                self.trialResponse.append(True)
                                self.trialResponseDir.append(moveDir)
                                self.trialResponseFrame.append(self._sessionFrame)
                                hasResponded = True
                    else:
                        # response window ended
                        self.trialResponse.append(False)
                        self.trialResponseDir.append(np.nan)
                        self.trialResponseFrame.append(self._sessionFrame)
                        hasResponded = True
                
                # show target
                if not self.isCatchTrial[-1]:
                    target.pos = targetPos 
                    target.draw()
            
            if hasResponded:
                if (not self.isCatchTrial[-1] and self.trialResponse[-1] > 0 and 
                    self._sessionFrame <= self.trialResponseFrame[-1] + max(self.rewardDelayFrames,self.postRewardTargetFrames)):
                    if self._sessionFrame <= self.trialResponseFrame[-1] + self.postRewardTargetFrames:
                        # hold target at reward position after correct trial
                        if self._sessionFrame == self.trialResponseFrame[-1]:
                            targetPos[0] = targetStartPosPix[0] + moveDir * rewardMove
                            target.pos = targetPos
                        target.draw()
                    if self._sessionFrame == self.trialResponseFrame[-1] + self.rewardDelayFrames:
                        # deliver reward
                        self._reward = self.solenoidOpenTime    
                else:
                    # end trial
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if self.maxTrials is not None and len(self.trialStartFrame) >= self.maxTrials:
                        self._continueSession = False
            
            self.showFrame()


def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))


if __name__ == "__main__":
    pass