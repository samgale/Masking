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


class FNTask(TaskControl):
    
    def __init__(self,rigName,taskVersion=None):
        TaskControl.__init__(self,rigName)
        
        self.taskVersion = taskVersion
        self.maxTrials = None
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 600 # max total preStim frames
        self.quiescentFrames = 60 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFrames = 18 # min frames after stimulus onset before wheel movement has effects
        self.maxResponseWaitFrames = 120 # max frames from stim onset to end of trial
        
        self.wheelRewardDistance = 8.0 # mm of wheel movement to achieve reward
        self.maxQuiescentMoveDist = 1.0 # max allowed mm of wheel movement during quiescent period
        
        self.targetStartPos = [-0.25,0] # normalized initial xy  position of target; center (0,0), bottom-left (-0.5,-0.5), top-right (0.5,0.5)
        self.targetRewardPos = [0.25,0]
        self.targetAutoMoveRate = 0 # fraction of normalized screen width per second that target automatically moves
        self.keepTargetOnScreen = True
        self.targetContrast = 1
        self.targetSize = 25 # degrees
        self.targetSF = 0.08 # cycles/deg
        self.targetOri = 0
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)

    
    def setDefaultParams(self,taskVersion):
        if taskVersion == 'training1':
            # stim moves to reward automatically; wheel movement ignored
            self.spacebarRewardsEnabled = False
            self.equalSampling = False
            self.probGoRight = 0.5
            self.probCatch = 0
            self.rewardCatchNogo = False
            self.targetContrast = [1]
            self.moveStim = True
            self.postRewardTargetFrames = 60
            self.preStimFramesFixed = 360
            self.preStimFramesVariableMean = 120
            self.preStimFramesMax = 600
            self.quiescentFrames = 0
            self.maxResponseWaitFrames = 3600
            self.useGoTone = False
            self.solenoidOpenTime = 0.2
            self.gratingEdge= 'raisedCos'
            if option in ('rot','rotation'):
                self.rotateTarget = True
                self.normTargetPos = [(0,0)]
                self.targetOri = [-45,45]
                self.autoRotationRate = 45
                self.rewardRotation = 45
                self.targetSize = 50
                self.gratingEdgeBlurWidth = 0.04
            else:
                self.rotateTarget = False
                self.normTargetPos = [(-0.25,0),(0.25,0)]
                self.targetOri = [0]
                self.normAutoMoveRate = 0.25
                self.normRewardDistance =  0.25
                self.targetSize = 25
                self.gratingEdgeBlurWidth = 0.08
            
        elif taskVersion == 'training2':
            # learn to associate wheel movement with stimulus movement and reward
            # only use 1-2 sessions
            self.setDefaultParams('training1',option)
            self.normAutoMoveRate = 0
            self.autoRotationRate = 0
            self.wheelRewardDistance = 4.0
            self.openLoopFramesFixed = 18
            self.openLoopFramesVariableMean = 0
            self.maxResponseWaitFrames = 3600
            self.useIncorrectNoise = False
            self.incorrectTimeoutFrames = 0
            self.incorrectTrialRepeats = 25   #avoid early bias formation
            self.solenoidOpenTime = 0.1
            
        elif taskVersion == 'training3':
            # introduce quiescent period, shorter response window, incorrect penalty, and catch trials
            self.setDefaultParams('training2',option)
            self.wheelRewardDistance = 3.0  # increase   
            self.quiescentFrames = 60
            self.maxResponseWaitFrames = 1200 # adjust this 
            self.useIncorrectNoise = True
            self.incorrectTimeoutFrames = 360
            self.incorrectTrialRepeats = 5 # will repeat for unanswered trials
            self.solenoidOpenTime = 0.07
            self.probCatch = 0.15
            
        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
    
     
    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create target stimulus
        targetStartPosPix,targetRewardPosPix = [[p * s for p,s in zip(pos,self.monSizePix)] for pos in (self.targetStartPos,self.targetRewardPos)]
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
        rewardDir = 1 if targetRewardPosPix > targetStartPosPix else -1
        rewardMove = abs(targetRewardPosPix - targetStartPosPis)
        self.wheelGain = rewardMove / (self.wheelRewardDistance / self.wheelRadius)
        maxQuiescentMove = (self.maxQuiescentMoveDist / self.wheelRadius) * self.wheelGain
        monitorEdge = 0.5 * (self.monSizePix[0] - targetSizePix)
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.quiescentMoveFrames = [] # frames where quiescent period was violated
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames) # can grow larger than preStimFrames during quiescent period
                quiescentWheelMove = 0 # virtual (not on screen) change in target position during quiescent period
                closedLoopWheelMove = 0 # actual change in target position during closed loop period
                targetPos = targetStartPosPix
                self.trialStartFrame.append(self._sessionFrame)
                hasResponded = False
            
            # extend pre stim gray frames if wheel moving during quiescent period
            if self.trialPreStimFrames[-1] - self.quiescentFrames < self._trialFrame < self.trialPreStimFrames[-1]:
                quiescentWheelMove += self.deltaWheelPos[-1] * self.wheelGain
                if abs(quiescentWheelMove) > maxQuiescentMove:
                    self.quiescentMoveFrames.append(self._sessionFrame)
                    self.trialPreStimFrames[-1] += randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                    quiescentWheelMove = 0
            
            # if gray screen period is complete but before response
            if not hasResponded and self._trialFrame >= self.trialPreStimFrames[-1]:
                if self._trialFrame == self.trialPreStimFrames[-1]:
                    self.trialStimStartFrame.append(self._sessionFrame)
                
                if self._trialFrame >= self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
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
                
                if self._trialFrame >= self.trialPreStimFrames[-1] :
                    target.pos = targetPos 
                    target.draw()
                
                if self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    if targetFrames > 0:
                        self.trialResponse.append(0) # no response on go trial
                        if self.useIncorrectNoise:
                            self._noise = True
                    elif rewardDir == 0:
                        self.trialResponse.append(1) # no response on no-go trial
                        self._reward = rewardSize
                    else:
                        self.trialResponse.append(np.nan) # no response on catch trial
                    self.trialResponseFrame.append(self._sessionFrame)
                    self.trialResponseDir.append(np.nan)
                    hasResponded = True
                elif abs(closedLoopWheelMove) > rewardMove:
                    moveDir = 1 if closedLoopWheelMove > 0 else -1
                    if not (self.keepTargetOnScreen and moveDir == -rewardDir):
                        if np.isnan(rewardDir):
                            self.trialResponse.append(np.nan) # movement on catch trial
                        elif moveDir == rewardDir:
                            self.trialResponse.append(1) # correct movement on go trial
                            self._reward = rewardSize
                        else:
                            if rewardDir == 0:
                                self.trialResponse.append(-1) # movement during no-go
                            else:
                                self.trialResponse.append(-1) # incorrect movement on go trial
                            if self.useIncorrectNoise:
                                self._noise = True
                        self.trialResponseFrame.append(self._sessionFrame)
                        self.trialResponseDir.append(moveDir)
                        hasResponded = True
                elif rewardDir == 0 and abs(closedLoopWheelMove) > maxQuiescentMove:
                    self.trialResponse.append(-1) # movement during no-go
                    if self.useIncorrectNoise:
                        self._noise = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    self.trialResponseDir.append(np.nan)
                    hasResponded = True
                
            # show any post response stimuli or end trial
            if hasResponded:
                if (self.moveStim and self.trialResponse[-1] > 0 and targetFrames > 0 and
                    self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames):
                    # hold target and reward pos/ori after correct trial
                    if self._sessionFrame == self.trialResponseFrame[-1]:
                        if self.rotateTarget:
                            targetOri = initTargetOri + rewardMove * rewardDir
                            target.ori = targetOri
                        else:
                            targetPos[0] = initTargetPos[0] + rewardMove * rewardDir
                            target.pos = targetPos
                    target.draw()
                elif (self.trialResponse[-1] < 1 and 
                      self._sessionFrame < self.trialResponseFrame[-1] + self.incorrectTimeoutFrames):
                    # wait for incorrectTimeoutFrames after incorrect trial
                    # if rotation task, hold target at incorrect ori for postRewardTargetFrames
                    if (self.rotateTarget and targetFrames > 0 and self.trialResponse[-1] < 0 and
                        self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames):
                        if self._sessionFrame == self.trialResponseFrame[-1]:
                            target.ori = initTargetOri + rewardMove * -rewardDir
                        target.draw()
                elif (self.showVisibilityRating or not np.isnan(optoOnset)) and self._trialFrame < self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    pass # wait until end of response window
                elif self.showVisibilityRating and visRating is None:
                    if len(self.visRatingStartFrame) < len(self.trialStartFrame):
                        self.visRatingStartFrame.append(self._sessionFrame)
                        self._mouse.clickReset()
                        if self.allowMouseClickVisRating:
                            self._mouse.setVisible(True)
                    ratingTitle.draw()
                    for button in ratingButtons:
                        button.draw()
                    for key in ('1','2','3'):
                        if key in self._keys:
                            visRating = key
                            break
                    else: # check for mouse click if no key press
                        if self.allowMouseClickVisRating and self._mouse.getPressed()[0]:
                            mousePos = self._mouse.getPos()
                            for button in ratingButtons:
                                if all([button.pos[i] - buttonSize < mousePos[i] < button.pos[i] + buttonSize for i in (0,1)]):
                                    visRating = button.text
                                    break
                else:
                    if self.showVisibilityRating:
                        self.visRating.append(visRating)
                        self.visRatingEndFrame.append(self._sessionFrame)
                        visRating = None
                        self._mouse.setVisible(False)
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if self.trialResponse[-1] < 1 and incorrectRepeatCount < self.incorrectTrialRepeats:
                        incorrectRepeatCount += 1
                        self.trialRepeat.append(True)
                    else:
                        incorrectRepeatCount = 0
                        self.trialRepeat.append(False)
                    if self.maxTrials is not None and len(self.trialStartFrame) >= self.maxTrials:
                        self._continueSession = False
            
            self.showFrame()


def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))


if __name__ == "__main__":
    pass