# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import math, random
from psychopy import visual    
from TaskControl import TaskControl


class MaskingTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        
        # parameters that can vary across trials are lists
        # only one of targetPos and targetOri can be len() > 1
        
        self.probNoGo = 0 # fraction of trials with no target, rewarded for no movement of wheel
        self.probGoRight = 0.5 # fraction of go trials rewarded for rightward movement of wheel
        self.probMask = 0 # fraction of trials with mask
        self.maxConsecutiveMaskTrials = 3
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 720 # max total preStim frames
        self.quiescentFrames = 60 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFramesFixed = 24 # min frames after stimulus onset before wheel movement has effects
        self.openLoopFramesVariableMean = 0 # mean of additional open loop frames drawn from exponential distribution
        self.openLoopFramesMax = 120 # max total openLoopFrames
        self.maxResponseWaitFrames = 360 # max frames between end of openLoopFrames and end of go trial
        self.nogoWaitFrames = 120 # frames after openLoopFrames during which mouse must remain still on nogo trials
        
        self.normRewardDistance = 0.25 # normalized to screen width
        self.gratingRotationGain = 0 # degrees per pixels of wheel movement
        self.rewardRotation = 45 # degrees
        self.maxQuiescentNormMoveDist = 0.025 # movement threshold during quiescent period
        self.useGoTone = False # play tone when openLoopFrames is complete
        self.useIncorrectNoise = False # play noise when trial is incorrect
        self.incorrectTrialRepeats = 0 # maximum number of incorrect trial repeats
        self.incorrectTimeoutFrames = 0 # extended gray screen following incorrect trial
        
        # mouse can move target stimulus with wheel for early training
        # varying stimulus duration and/or masking not part of this stage
        self.moveStim = False
        self.normAutoMoveRate = 0 # fraction of screen width per second that target automatically moves
        self.autoRotationRate = 0 # deg/s
        self.keepTargetOnScreen = False # false allows for incorrect trials during training
        self.reversePhasePeriod = 0 # frames
        self.gratingDriftFreq = 0 # cycles/s
        self.postRewardTargetFrames = 1 # frames to freeze target after reward
        
        # target stimulus params
        self.normTargetPos = [(0,0)] # normalized initial xy  position of target; center (0,0), bottom-left (-0.5,-0.5), top-right (0.5,0.5)
        self.targetFrames = [2] # duration of target stimulus
        self.targetContrast = [1]
        self.targetSize = 20 # degrees
        self.targetSF = 0.1 # cycles/deg
        self.targetOri = [-45,45] # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'circle' # 'gauss' or 'circle'
        
        # mask params
        self.maskType = None # None, 'plaid', or 'noise'
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskFrames = 9 # duration of mask
        self.maskOnset = [30] # frames >=0 relative to target stimulus onset
        self.maskContrast = [1]     

    
    def setDefaultParams(self,name,taskVersion=None):
        if name == 'training1':
            # stim moves to reward automatically; wheel movement ignored
            self.probGoRight = 0.5
            self.probNoGo = 0
            self.moveStim = True
            self.maxResponseWaitFrames = 3600
            self.postRewardTargetFrames = 60
            self.useGoTone = True
            self.preStimFramesFixed = 360
            self.preStimFramesVariableMean = 120
            self.preStimFramesMax = 600
            self.quiescentFrames = 0
            self.openLoopFramesFixed = 24
            self.openLoopFramesVariableMean = 0
            self.gratingEdge = 'circle'
            if taskVersion in ('rot','rotation'):
                self.normTargetPos = [(0,0)]
                self.targetOri = [-45,45]
                self.autoRotationRate = 45
                self.gratingRotationGain = 0.05
                self.rewardRotation = 45
                self.targetSize = 48.5
            else:
                if taskVersion in ('pos','position'):
                    self.normTargetPos = [(-0.25,0),(0.25,0)]
                    self.targetOri = [0]
                else:
                    self.normTargetPos = [(0,0)]
                    self.targetOri = [-45,45]
                self.normAutoMoveRate = 0.25
                self.normRewardDistance =  0.25
                self.targetSize = 28
            
        elif name == 'training2':
            # learning to associate wheel movement with stimulus movement and reward
            # only use 1-2 sessions
            self.setDefaultParams('training1',taskVersion)
            self.normAutoMoveRate = 0
            self.keepTargetOnScreen=False
            self.normRewardDistance = 0.15 
            self.maxResponseWaitFrames = 3600
            self.incorrectTimeoutFrames = 240
            self.useIncorrectNoise=False
            self.solenoidOpenTime = 0.07
            self.incorrectTrialRepeats = 5  # will repeat for unanswered trials 
            if taskVersion in ('rot','rotation'):
                self.autoRotationRate = 0  
                self.useGoTone = False
            
        elif name == 'training3':
            # start training, introduce incorrect trials and shorter wait time
            self.setDefaultParams('training2',taskVersion)
            self.normRewardDistance = 0.18
            self.maxResponseWaitFrames = 720   # manually adjust this 
            self.incorrectTrialRepeats = 30
            self.useIncorrectNoise = True
            self.quiescentFrames = 60
            
        elif name == 'training4':
            # similar to training3 but more stringent parameter settings, add q period
            self.setDefaultParams('training3',taskVersion)
            self.normRewardDistance = 0.2
            self.maxResponseWaitFrames = 120
            self.incorrectTrialRepeats = 20
            self.incorrectTimeoutFrames = 600
            self.solenoidOpenTime = 0.05
            
        elif name == 'training5':
            # introduce no-go trials
            self.setDefaultParams('training4',taskVersion)
            self.normRewardDistance = 0.22
            self.maxResponseWaitFrames = 60
            self.probNoGo = 0.33
            self.incorrectTrialRepeats = 100
            self.incorrectTimeoutFrames = 720
            
        elif name == 'training6':
            # introduce variable open loop frames
            self.setDefaultParams('training5',taskVersion)
            self.maxResponseWaitFrames = 60
            self.openLoopFramesFixed = 24
            self.openLoopFramesVariableMean = 36
            self.openLoopFramesMax = 180
            self.incorrectTrialRepeats = 0
            
        else:
            print(str(name)+' is not a recognized set of default parameters')
    
     
    def checkParamValues(self):
        assert((len(self.normTargetPos)>1 and len(self.targetOri)==1) or
               (len(self.normTargetPos)==1 and len(self.targetOri)>1))
        assert(self.quiescentFrames <= self.preStimFramesFixed)
        assert(0 not in self.targetFrames + self.targetContrast + self.maskOnset + self.maskContrast)
        for prob in (self.probNoGo,self.probGoRight,self.probMask):
            assert(0 <= prob <= 1)
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create target stimulus
        targetPosPix = [tuple(p[i] * self.monSizePix[i] for i in (0,1)) for p in self.normTargetPos]
        targetSizePix = int(self.targetSize * self.pixelsPerDeg)
        sf = self.targetSF / self.pixelsPerDeg
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask=self.gratingEdge,
                                    tex=self.gratingType,
                                    size=targetSizePix, 
                                    sf=sf)  
        
        # create mask
        # 'target' mask overlaps target
        # 'surround' mask surrounds but does not overlap target
        # 'full' mask surrounds and overlaps target
        if self.maskShape=='target':
            maskPos = targetPosPix
            maskSize = targetSizePix
            maskEdgeBlur = self.gratingEdge
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
                                       tex=self.gratingType,
                                       size=maskSize,
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
                                     size=maskSize)
                                     for pos in maskPos]
        
        if self.maskShape=='surround':
            mask += [visual.Circle(win=self._win,
                                   units='pix',
                                   radius=0.5*targetSizePix,
                                   lineColor=0.5,
                                   fillColor=0.5)
                                   for pos in targetPosPix]
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialOpenLoopFrames = []
        self.trialTargetPos = []
        self.trialTargetOri = []
        self.trialTargetContrast = []
        self.trialTargetFrames = []
        self.trialMaskOnset = []
        self.trialMaskContrast = []
        self.trialRewardDir = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.trialRepeat = [False]
        self.quiescentMoveFrames = [] # frames where quiescent period was violated
               
        monitorEdge = 0.5 * (self.monSizePix[0] - targetSizePix)
        maxQuiescentMove = self.maxQuiescentNormMoveDist * self.monSizePix[0]
        rotateTarget = True if len(self.targetOri) > 1 and (self.gratingRotationGain > 0 or self.autoRotationRate > 0) else False
        if rotateTarget:
            maxQuiescentMove *= self.gratingRotationGain
            rewardMove = self.rewardRotation
        else:
            rewardMove = self.normRewardDistance * self.monSizePix[0]
        incorrectRepeatCount = 0
        maskCount = 0
        
        while self._continueSession: # each loop is a frame presented on the monitor
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
                    showMask = random.random() < self.probMask if len(self.trialResponse) > 0 and maskCount < self.maxConsecutiveMaskTrials else False
                    maskCount = maskCount + 1 if showMask else 0
                    if random.random() < self.probNoGo:
                        rewardDir = 0
                        initTargetPos = (0,0)
                        initTargetOri = 0
                        targetContrast = 0
                        targetFrames = 0
                        maskOnset = 0
                        maskContrast = random.choice(self.maskContrast) if showMask else 0
                    else:
                        if showMask:
                            maskOnset = random.choice(self.maskOnset+[0]) if rotateTarget else random.choice(self.maskOnset)
                            maskContrast = random.choice(self.maskContrast)
                        else:
                            maskOnset = maskContrast = 0
                        if rotateTarget and maskOnset == 0 and maskContrast > 0:
                            rewardDir = 0
                            initTargetPos = (0,0)
                            initTargetOri = 0
                            targetContrast = 0
                            targetFrames = 0
                        else:
                            goRight = random.random() < self.probGoRight
                            rewardDir = 1 if goRight else -1
                            if len(targetPosPix) > 1:
                                if goRight:
                                    initTargetPos = random.choice([pos for pos in targetPosPix if pos[0] < 0])
                                else:
                                    initTargetPos = random.choice([pos for pos in targetPosPix if pos[0] > 0])
                                initTargetOri = self.targetOri[0]
                            else:
                                initTargetPos = targetPosPix[0]
                                if (rotateTarget and goRight) or (not rotateTarget and not goRight):
                                    initTargetOri = random.choice([ori for ori in self.targetOri if ori < 0])
                                else:
                                    initTargetOri = random.choice([ori for ori in self.targetOri if ori > 0])
                            targetContrast = random.choice(self.targetContrast)
                            targetFrames = random.choice(self.targetFrames)
                
                targetPos = list(initTargetPos) # position of target on screen
                targetOri = initTargetOri # orientation of target on screen
                target.pos = targetPos
                target.ori = targetOri
                target.contrast = targetContrast
                target.phase = (0,0)
                if self.maskType is not None:
                    for m in mask:
                        m.contrast = maskContrast
                        if self.maskType == 'noise' and isinstance(m,visual.NoiseStim):
                            m.updateNoise()
                self.trialStartFrame.append(self._sessionFrame)
                self.trialTargetPos.append(initTargetPos)
                self.trialTargetOri.append(initTargetOri)
                self.trialTargetContrast.append(targetContrast)
                self.trialTargetFrames.append(targetFrames)
                self.trialMaskOnset.append(maskOnset)
                self.trialMaskContrast.append(maskContrast)
                self.trialRewardDir.append(rewardDir)
                hasResponded = False
            
            # extend pre stim gray frames if wheel moving during quiescent period
            if self.trialPreStimFrames[-1] - self.quiescentFrames < self._trialFrame < self.trialPreStimFrames[-1]:
                quiescentWheelMove += self.deltaWheelPos[-1] * self.gratingRotationGain if rotateTarget else self.deltaWheelPos[-1]
                if abs(quiescentWheelMove) > maxQuiescentMove:
                    self.quiescentMoveFrames.append(self._sessionFrame)
                    self.trialPreStimFrames[-1] += preStimFrames
                    quiescentWheelMove = 0
            
            # if gray screen period is complete, update target and mask stimuli
            if not hasResponded and self._trialFrame >= self.trialPreStimFrames[-1]:
                if self._trialFrame == self.trialPreStimFrames[-1]:
                    self.trialStimStartFrame.append(self._sessionFrame)
                if self._trialFrame >= self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
                    if self.useGoTone and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
                        self._tone = True
                    if self.moveStim:
                        if rotateTarget:
                            if self.autoRotationRate > 0:
                                deltaOri = rewardDir * self.autoRotationRate / self.frameRate
                                targetOri += deltaOri
                                closedLoopWheelMove += deltaOri
                            elif self.gratingRotationGain > 0:
                                deltaOri = self.deltaWheelPos[-1] * self.gratingRotationGain
                                targetOri += deltaOri
                                closedLoopWheelMove += deltaOri
                                if self.keepTargetOnScreen and abs(targetOri) > 90:
                                    adjust = targetOri - 90 if targetOri > 90 else targetOri + 90
                                    targetOri -= adjust
                                    closedLoopWheelMove -= adjust
                            target.ori = targetOri
                        else:
                            if self.normAutoMoveRate > 0:
                                deltaPos = rewardDir * self.normAutoMoveRate * self.monSizePix[0] / self.frameRate
                                targetPos[0] += deltaPos
                                closedLoopWheelMove += deltaPos
                            else:
                                targetPos[0] += self.deltaWheelPos[-1]
                                closedLoopWheelMove += self.deltaWheelPos[-1]
                            if self.keepTargetOnScreen and abs(targetPos[0]) > monitorEdge:
                                adjust = targetPos[0] - monitorEdge if targetPos[0] > 0 else targetPos[0] + monitorEdge
                                targetPos[0] -= adjust
                                closedLoopWheelMove -= adjust
                            target.pos = targetPos 
                    else:
                        closedLoopWheelMove += self.deltaWheelPos[-1] * self.gratingRotationGain if rotateTarget else self.deltaWheelPos[-1]
                if self.moveStim:
                    if targetFrames > 0:
                        if self.gratingDriftFreq > 0:
                            target.phase[0] += rewardDir * self.gratingDriftFreq / self.frameRate
                            target.phase = target.phase
                        elif self.reversePhasePeriod > 0 and ((self._trialFrame - self.trialPreStimFrames[-1]) % self.reversePhasePeriod) == 0:
                            phase = (0.5,0) if target.phase[0] == 0 else (0,0)
                            target.phase = phase
                        target.draw()
                else:
                    if (self.maskType is not None and maskContrast > 0 and
                        (self.trialPreStimFrames[-1] + maskOnset <= self._trialFrame < 
                         self.trialPreStimFrames[-1] + maskOnset + self.maskFrames)):
                        for m in mask:
                            m.draw()
                    elif self._trialFrame < self.trialPreStimFrames[-1] + targetFrames:
                        target.draw()
            
                # define response if wheel moved past threshold (either side) or max trial duration reached
                # trialResponse for go trials is 1 for correct direction, -1 for incorrect direction, or 0 for no response
                # trialResponse for no go trials is 1 for no response or -1 for movement in either direction
                if targetFrames > 0 and closedLoopWheelMove * rewardDir > rewardMove:
                    self.trialResponse.append(1) # correct movement
                    self._reward = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                elif ((targetFrames == 0 and abs(closedLoopWheelMove) > maxQuiescentMove) or
                      (not self.keepTargetOnScreen and closedLoopWheelMove * -rewardDir > rewardMove)):
                    self.trialResponse.append(-1) # incorrect movement
                    if self.useIncorrectNoise and not (rotateTarget and targetFrames == 0):
                        self._noise = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                elif targetFrames > 0 and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    self.trialResponse.append(0) # no response on go trial
                    if self.useIncorrectNoise:
                        self._noise = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                elif targetFrames==0 and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.nogoWaitFrames:
                    self.trialResponse.append(1) # correct no response
                    if not rotateTarget:
                        self._reward = True  
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                
            # show any post response stimuli or end trial
            if hasResponded:
                if (self.trialResponse[-1] > 0 and targetFrames > 0 and
                    self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames):
                    # hold target and reward pos/ori after correct trial
                    if self._sessionFrame == self.trialResponseFrame[-1]:
                        if rotateTarget:
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
                    if (rotateTarget and targetFrames > 0 and self.trialResponse[-1] < 0 and
                        self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames):
                        if self._sessionFrame == self.trialResponseFrame[-1]:
                            target.ori = initTargetOri + rewardMove * -rewardDir
                        target.draw()
                else:
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if self.trialResponse[-1] < 1 and not (rotateTarget and targetFrames ==0) and incorrectRepeatCount < self.incorrectTrialRepeats:
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