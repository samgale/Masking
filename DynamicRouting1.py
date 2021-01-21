# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import random
from psychopy import visual    
from TaskControl import TaskControl


class DynamicRouting1(TaskControl):
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        
        self.trialsPerBlock = [1,1] # min and max trials per block
        self.rewardBothDirs = False
        self.blockProbGoRight = [0,1] # fraction of trials in block rewarded for rightward movement of wheel
        self.probCatch = 0 # fraction of trials with no target and no reward
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 600 # max total preStim frames
        self.quiescentFrames = 0 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFramesFixed = 18 # min frames after stimulus onset before wheel movement has effects
        self.openLoopFramesVariableMean = 0 # mean of additional open loop frames drawn from exponential distribution
        self.openLoopFramesMax = 120 # max total openLoopFrames
        self.maxResponseWaitFrames = 120 # max frames between end of openLoopFrames and end of go trial
        
        self.rewardSizeLeft = self.rewardSizeRight = None # set solenoid open time in seconds; otherwise defaults to self.solenoidOpenTime
        self.wheelRewardDistance = 8.0 # mm of wheel movement to achieve reward
        self.maxQuiescentMoveDist = 1.0 # max allowed mm of wheel movement during quiescent period
        self.useGoTone = False # play tone when openLoopFrames is complete
        self.useIncorrectNoise = False # play noise when trial is incorrect
        self.incorrectTrialRepeats = 0 # maximum number of incorrect trial repeats
        self.incorrectTimeoutFrames = 0 # extended gray screen following incorrect trial
        
        # mouse can move target stimulus with wheel for early training
        self.moveStim = False
        self.normAutoMoveRate = 0 # fraction of screen width per second that target automatically moves
        self.postRewardTargetFrames = 1 # frames to freeze target after reward
        
        # target stimulus params
        # parameters that can vary across trials are lists
        self.targetFrames = [6] # duration of target stimulus
        self.targetContrast = [1]
        self.targetSize = 25 # degrees
        self.targetSF = 0.08 # cycles/deg
        self.targetOri = 0 # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos

    
    def setDefaultParams(self,name,taskVersion=None):
        if name == 'training0':
            # stim moves to reward automatically; wheel movement ignored
            self.moveStim = True
            self.normAutoMoveRate = 0.5
            self.maxResponseWaitFrames = 3600
            self.blockProbGoRight = [0.5]
            self.rewardBothDirs = True
            self.postRewardTargetFrames = 60
            
        elif name == 'training1':
            # learn to associate wheel movement with stimulus movement and reward
            # either diretion rewarded
            self.setDefaultParams('training0',taskVersion)
            self.normAutoMoveRate = 0
            
        elif name == 'training2':
            # one side rewarded
            # introduce quiescent period, shorter response window, incorrect repeats, and catch trials
            self.setDefaultParams('training1',taskVersion)
            self.rewardBothDirs = False 
            self.quiescentFrames = 60
            self.maxResponseWaitFrames = 1200 # adjust this 
            self.useIncorrectNoise = True
            self.incorrectTimeoutFrames = 360
            self.incorrectTrialRepeats = 5 # will repeat for unanswered trials
            self.probCatch = 0.1
            
        elif name == 'training3':
            # introduce block structure
            self.setDefaultParams('training2',taskVersion)
            self.trialsPerBlock = [3,8]
            self.blockProbGoRight = [0,1]
            
        else:
            print(str(name)+' is not a recognized set of default parameters')
    
     
    def checkParamValues(self):
        assert(self.quiescentFrames <= self.preStimFramesFixed)
        assert(self.maxQuiescentMoveDist <= self.wheelRewardDistance) 
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create target stimulus
        targetSizePix = int(self.targetSize * self.pixelsPerDeg)
        sf = self.targetSF / self.pixelsPerDeg
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask=self.gratingEdge,
                                    maskParams=edgeBlurWidth,
                                    tex=self.gratingType,
                                    size=targetSizePix, 
                                    sf=sf,
                                    ori=self.targetOri)
            
        # calculate pixels to move stimulus per radian of wheel movement
        rewardMove = 0.5 * (self.monSizePix[0] - targetSizePix)
        self.wheelGain = rewardMove / (self.wheelRewardDistance / self.wheelRadius)
        maxQuiescentMove = (self.maxQuiescentMoveDist / self.wheelRadius) * self.wheelGain
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialOpenLoopFrames = []
        self.trialTargetContrast = []
        self.trialTargetFrames = []
        self.trialRewardDir = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.trialRewarded = []
        self.trialRepeat = [False]
        self.quiescentMoveFrames = [] # frames where quiescent period was violated
        self.trialBlock = []
        self.trialProbGoRight = []
        blockTrials = None # number of trials of current block
        blockTrialCount = None # number of trials completed in current block
        probGoRight = None # probability that rightward movement rewarded in current block
        incorrectRepeatCount = 0
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames) # can grow larger than preStimFrames during quiescent period
                self.trialOpenLoopFrames.append(randomExponential(self.openLoopFramesFixed,self.openLoopFramesVariableMean,self.openLoopFramesMax))
                quiescentWheelMove = 0 # virtual (not on screen) change in target position/ori during quiescent period
                closedLoopWheelMove = 0 # actual or virtual change in target position/ori during closed loop period
                
                if not self.trialRepeat[-1]:
                    if blockTrials is not None and random.random() < self.probCatch:
                        rewardDir = targetContrast = targetFrames = 0
                    else:
                        if blockTrials is None or blockTrialCount == blockTrials:
                            blockTrials = random.randint(*self.trialsPerBlock)
                            blockTrialCount = 1
                            probGoRight = self.blockProbGoRight[0] if len(self.blockProbGoRight) == 1 else random.choice([p for p in self.blockProbGoRight if p != probGoRight])
                            if len(self.trialBlock) < 1:
                                self.trialBlock.append(0)
                            else:
                                self.trialBlock.append(self.trialBlock[-1] + 1)
                        else:
                            blockTrialCount += 1
                            
                        rewardDir = 1 if random.random() < probGoRight else -1
                        targetContrast = random.choice(self.targetContrast)
                        targetFrames = random.choice(self.targetFrames)
                    
                        if rewardDir == 1 and self.rewardSizeRight is not None:
                            rewardSize = self.rewardSizeRight
                        elif rewardDir == -1 and self.rewardSizeLeft is not None:
                            rewardSize = self.rewardSizeLeft
                        else:
                            rewardSize = self.solenoidOpenTime
                
                targetPos = [0,0] # position of target on screen
                target.pos = targetPos
                target.contrast = targetContrast

                self.trialStartFrame.append(self._sessionFrame)
                self.trialRewardDir.append(rewardDir)
                self.trialTargetContrast.append(targetContrast)
                self.trialTargetFrames.append(targetFrames)
                if blockTrialCount > 1:
                    self.trialBlock.append(self.trialBlock[-1])
                self.trialProbGoRight.append(probGoRight)
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
                    if self.useGoTone and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
                        self._tone = True
                    if self.moveStim:
                        if self.normAutoMoveRate > 0:
                            deltaPos = rewardDir * self.normAutoMoveRate * self.monSizePix[0] * self._win.monitorFramePeriod
                            targetPos[0] += deltaPos
                            closedLoopWheelMove += deltaPos
                        else:
                            deltaPos = self.deltaWheelPos[-1] * self.wheelGain
                            targetPos[0] += deltaPos
                            closedLoopWheelMove += deltaPos
                        target.pos = targetPos 
                    else:
                        closedLoopWheelMove += self.deltaWheelPos[-1] * self.wheelGain
                if (self.moveStim and rewardDir != 0) or self._trialFrame < self.trialPreStimFrames[-1] + targetFrames:
                        target.draw()
                    
                # define response if wheel moved past threshold (either side) or max trial duration reached
                if self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    self.trialResponse.append(0) # no response
                    self.trialResponseFrame.append(self._sessionFrame)
                    self.trialRewarded.append(False)
                    if self.useIncorrectNoise and rewardDir != 0:
                        self._noise = True
                    hasResponded = True
                elif abs(closedLoopWheelMove) > rewardMove:
                    moveDir = 1 if closedLoopWheelMove > 0 else -1
                    self.trialResponse.append(moveDir)
                    self.trialResponseFrame.append(self._sessionFrame)
                    if self.rewardBothDirs or moveDir == rewardDir:
                        self.trialRewarded.append(True)
                        self._reward = rewardSize
                    else:
                        self.trialRewarded.append(False)
                        if moveDir != 0 and self.useIncorrectNoise:
                            self._noise = True
                    hasResponded = True
                
            # show any post response stimuli or end trial
            if hasResponded:
                if self.moveStim and self.trialRewarded[-1] and self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames:
                    # hold target and reward pos/ori after correct trial
                    if self._sessionFrame == self.trialResponseFrame[-1]:
                        targetPos[0] = rewardMove * moveDir
                        target.pos = targetPos
                    target.draw()
                elif (rewardDir != 0 and not self.trialRewarded[-1] and 
                      self._sessionFrame < self.trialResponseFrame[-1] + self.incorrectTimeoutFrames):
                    # wait for incorrectTimeoutFrames after incorrect trial
                    pass
                else:
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if rewardDir != 0 and not self.trialRewarded[-1] and incorrectRepeatCount < self.incorrectTrialRepeats:
                        incorrectRepeatCount += 1
                        self.trialRepeat.append(True)
                    else:
                        incorrectRepeatCount = 0
                        self.trialRepeat.append(False)
            
            self.showFrame()


def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))


if __name__ == "__main__":
    pass