# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools, math, random
from psychopy import visual    
from TaskControl import TaskControl


class MaskingTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        
        # parameters that can vary across trials are lists
        # only one of targetPos and targetOri can be len() > 1
        
        self.taskVersion = None
        self.fracTrialsGoRight = 0.5 # if taskVersion defined, fraction of trials with go-right target, rewarded for rightward movement of wheel
        self.fracTrialsNoGo = 0 # fraction of trials with no target, rewarded for no movement of wheel
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 720 # max total preStim frames
        self.quiescentFrames = 60 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFramesFixed = 24 # min frames after stimulus onset before wheel movement has effects
        self.openLoopFramesVariableMean = 0 # mean of additional open loop frames drawn from exponential distribution
        self.openLoopFramesMax = 120 # max total openLoopFrames
        self.maxResponseWaitFrames = 360 # max frames between end of openLoopFrames and end of trial
        
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
        self.targetFrames = [2] # duration of target stimulus; targetFrames=0 for no response rewarded trials
        self.targetContrast = [1]
        self.targetSize = 45 # degrees
        self.targetSF = 0.04 # cycles/deg
        self.targetOri = [-45,45] # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'circle' # 'gauss' or 'circle'
        
        # mask params
        self.maskType = None # None, 'plaid', or 'noise'
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskFrames = 9 # duration of mask
        self.maskOnset = [30] # frames >=0 relative to target stimulus onset; include 0 for mask only trials
        self.maskContrast = [1] # include 0 for target only trials      

    
    def setDefaultParams(self,name):
        if name == 'training1':
            # stim moves to reward automatically; wheel movement ignored
            self.fracTrialsGoRight = 0.5
            self.fracTrialsNoGo = 0
            self.moveStim = True
            self.maxResponseWaitFrames = 3600
            self.postRewardTargetFrames = 60
            self.useGoTone = True
            self.preStimFramesFixed = 360
            self.preStimFramesVariableMean = 120
            self.preStimFramesMax = 720
            self.quiescentFrames = 0
            self.openLoopFramesFixed = 24
            self.openLoopFramesVariableMean = 0
            self.gratingEdge = 'circle'
            if self.taskVersion in ('rot','rotation'):
                self.autoRotationRate = 45
                self.rewardRotation = 45
                self.targetSize = 120
            else:
                self.normAutoMoveRate = 0.25
                self.normRewardDistance =  0.25
                self.targetSize = 60
            
        elif name == 'training2':
            # learning to associate wheel movement with stimulus movement and reward
            # only use 1-2 sessions
            self.setDefaultParams('training1')
            self.normAutoMoveRate = 0
            self.keepTargetOnScreen = True
            self.normRewardDistance = 0.12 
            self.maxResponseWaitFrames = 3600
            self.incorrectTimeoutFrames = 0
            self.incorrectTrialRepeats = 0
            
        elif name == 'training3':
            # start training, introduce incorrect trials and shorter wait time
            self.setDefaultParams('training2')
            self.normRewardDistance = 0.15
            self.maxResponseWaitFrames = 720
            self.keepTargetOnScreen = False
            self.incorrectTrialRepeats = 10
            
        elif name == 'training4':
            # similar to training3 but more stringent parameter settings, add q period
            self.setDefaultParams('training3')
            self.normRewardDistance = 0.18
            self.maxResponseWaitFrames = 120
            self.incorrectTrialRepeats = 6
            # self.incorrectTimeoutFrames = 240
            self.quiescentFrames = 60
            
        elif name == 'training5':
            # introduce no-go trials
            self.setDefaultParams('training4')
            self.normRewardDistance = 0.25
            self.maxResponseWaitFrames = 60
            self.fracTrialsGoRight = 0.33
            self.fracTrialsNoGo = 0.34
            self.incorrectTrialRepeats = 100
            
        elif name == 'training6':
            # introduce variable open loop frames
            self.setDefaultParams('training5')
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
        assert(0 not in self.targetFrames+self.maskOnset+self.maskContrast)
        assert((self.fracTrialsGoRight + self.fracTrialsNoGo) <= 1)
        

    def taskFlow(self):
        self.checkParamValues()
        
        # set up go right or left trials in correct proportions
        R = int(self.fracTrialsGoRight * 100)
        L = 100 - R - int(self.fracTrialsNoGo *100)
        if self.taskVersion in ('pos','position'):
            self.targetOri = [0]
            self.normTargetPos = [(-0.25,0)]*R + [(0.25,0)]*L
            self.gratingRotationGain = 0
        elif self.taskVersion in ('ori','orientatation'):
            self.targetOri = [-45]*L + [45]*R
            self.normTargetPos = [(0,0)]
            self.gratingRotationGain = 0
        elif self.taskVersion in ('rot','rotation'):
            self.targetOri = [-45]*R + [45]*L
            self.normTargetPos = [(0,0)]
            assert((self.gratingRotationGain > 0) or (self.autoRotationRate > 0))
        else:
            print(str(self.taskVersion)+' is not a recognized version of this task')
        
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
        
        # create list of trial parameter combinations
        # no mask trials (maskContrast=0)
        trialParams = list(itertools.product(targetPosPix,
                                             self.targetContrast,
                                             self.targetOri,
                                             self.targetFrames,
                                             [0],
                                             [0]))
        
        # add masking trials
        if self.maskType is not None:
            trialParams += list(itertools.product(targetPosPix,
                                                  self.targetContrast,
                                                  self.targetOri,
                                                  self.targetFrames,
                                                  self.maskOnset,
                                                  self.maskContrast))
        
        # add no response rewarded trials
        # includes trials with no target (targetFrames=0) and either no mask (maskContrast=0) or mask only (maskContrast>0)
        nogoMaskContrast = [] if self.maskType is None else self.maskContrast
        n = int(self.fracTrialsNoGo * len(trialParams) / (1 - self.fracTrialsNoGo) / (len(nogoMaskContrast)+1))
        trialParams += n * list(itertools.product([(0,0)],
                                                  [0],
                                                  [0],
                                                  [0],
                                                  [0],
                                                  [0]+nogoMaskContrast))
        
        random.shuffle(trialParams)
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialOpenLoopFrames = []
        self.trialTargetPos = []
        self.trialTargetContrast = []
        self.trialTargetOri = []
        self.trialTargetFrames = []
        self.trialMaskOnset = []
        self.trialMaskContrast = []
        self.trialRewardDir = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.quiescentMoveFrames = []   # frames where quiescent period was violated
        
        trialIndex = 0 # index of trialParams        
        monitorEdge = 0.5 * (self.monSizePix[0] - targetSizePix)
        maxQuiescentDist = self.maxQuiescentNormMoveDist * self.monSizePix[0]
        rewardDist = self.normRewardDistance * self.monSizePix[0]
        incorrectRepeatCount = 0
        
        while self._continueSession: # each loop is a frame presented on the monitor
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames)
                self.trialOpenLoopFrames.append(randomExponential(self.openLoopFramesFixed,self.openLoopFramesVariableMean,self.openLoopFramesMax))
                quiescentWheelPos = 0
                closedLoopWheelPos = 0
                initTargetPos,targetContrast,targetOri,targetFrames,maskOnset,maskContrast = trialParams[trialIndex]
                targetPos = list(initTargetPos)
                if len(self.normTargetPos) > 1:
                    rewardDir = -1 if targetPos[0] > 0 else 1
                elif self.gratingRotationGain > 0:
                    rewardDir = 1 if targetOri < 0 else -1
                else:
                    rewardDir = -1 if targetOri < 0 else 1
                target.pos = targetPos
                target.contrast = targetContrast
                target.ori = targetOri
                target.phase = (0,0)
                if self.maskType is not None:
                    for m in mask:
                        m.contrast = maskContrast
                        if self.maskType == 'noise' and isinstance(m,visual.NoiseStim):
                            m.updateNoise()
                self.trialStartFrame.append(self._sessionFrame)
                self.trialTargetPos.append(initTargetPos)
                self.trialTargetContrast.append(targetContrast)
                self.trialTargetOri.append(targetOri)
                self.trialTargetFrames.append(targetFrames)
                self.trialMaskOnset.append(maskOnset)
                self.trialMaskContrast.append(maskContrast)
                self.trialRewardDir.append(rewardDir)
                hasResponded = False
            
            # extend pre stim gray frames if wheel moving during quiescent period
            if self.trialPreStimFrames[-1] - self.quiescentFrames < self._trialFrame < self.trialPreStimFrames[-1]:
                quiescentWheelPos += self.deltaWheelPos[-1]
                if abs(quiescentWheelPos) > maxQuiescentDist:
                    self.quiescentMoveFrames.append(self._sessionFrame)
                    self.trialPreStimFrames[-1] += preStimFrames
                    quiescentWheelPos = 0
            
            # if gray screen period is complete, update target and mask stimuli
            if not hasResponded and self._trialFrame >= self.trialPreStimFrames[-1]:
                if self._trialFrame == self.trialPreStimFrames[-1]:
                    self.trialStimStartFrame.append(self._sessionFrame)
                if self._trialFrame >= self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
                    if self.useGoTone and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1]:
                        self._tone = True
                    if self.gratingRotationGain == 0:
                        if self.moveStim:
                            if self.autoRotationRate > 0:
                                target.ori = target.ori + rewardDir * self.autoRotationRate / self.frameRate
                            else:
                                if self.normAutoMoveRate > 0:
                                    autoDriftPix = rewardDir * self.normAutoMoveRate * self.monSizePix[0] / self.frameRate
                                    targetPos[0] += autoDriftPix
                                    closedLoopWheelPos += autoDriftPix
                                else:
                                    targetPos[0] += self.deltaWheelPos[-1]
                                    closedLoopWheelPos += self.deltaWheelPos[-1]
                                if self.keepTargetOnScreen and abs(targetPos[0]) > monitorEdge:
                                    adjust = targetPos[0] - monitorEdge if targetPos[0] > 0 else targetPos[0] + monitorEdge
                                    targetPos[0] -= adjust
                                    closedLoopWheelPos -= adjust
                                target.pos = targetPos
                        else:
                            closedLoopWheelPos += self.deltaWheelPos[-1]
                if self.moveStim:
                    if targetFrames > 0:
                        if self.gratingRotationGain > 0:
                            target.ori = target.ori + self.deltaWheelPos[-1] * self.gratingRotationGain
                            if self.keepTargetOnScreen:
                                if target.ori < -90:
                                    target.ori = -90
                                elif target.ori > 90:
                                    target.ori = 90
                        elif self.gratingDriftFreq > 0:
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
                if (targetFrames > 0 and ((closedLoopWheelPos * rewardDir > rewardDist) or 
                                         ((target.ori - targetOri) * rewardDir > self.rewardRotation))):
                    self.trialResponse.append(1) # correct movement
                    self._reward = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                elif ((targetFrames == 0 and abs(closedLoopWheelPos) > maxQuiescentDist) or
                      not self.keepTargetOnScreen and ((closedLoopWheelPos * -rewardDir > rewardDist) or
                                                      ((target.ori -targetOri) *-rewardDir > self.rewardRotation))):
                    self.trialResponse.append(-1) # incorrect movement
                    if self.useIncorrectNoise:
                        self._noise = True
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                elif not hasResponded and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    if targetFrames == 0:
                        self.trialResponse.append(1) # correct no response
                        self._reward = True
                    else:
                        self.trialResponse.append(0) # no response on go trial
                    self.trialResponseFrame.append(self._sessionFrame)
                    hasResponded = True
                
            # show any post response stimuli or end trial
            if hasResponded:
                if (self.trialResponse[-1] > 0 and targetFrames > 0 and
                    self._sessionFrame < self.trialResponseFrame[-1] + self.postRewardTargetFrames):
                    if self._sessionFrame == self.trialResponseFrame[-1]:
                        if self.gratingRotationGain > 0:
                            target.ori = targetOri + self.rewardRotation * rewardDir
                        else:
                            targetPos[0] = initTargetPos[0] + rewardDist * rewardDir
                            target.pos = targetPos
                    target.draw()
                elif (self.trialResponse[-1] < 1 and 
                      self._sessionFrame < self.trialResponseFrame[-1] + self.incorrectTimeoutFrames):
                    pass
                else:
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if self.trialResponse[-1] < 1 and incorrectRepeatCount < self.incorrectTrialRepeats:
                        incorrectRepeatCount += 1
                    else:
                        trialIndex += 1
                        incorrectRepeatCount = 0 
                    if trialIndex == len(trialParams):
                        trialIndex = 0
                        random.shuffle(trialParams)
            
            self.showFrame()


def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))


if __name__ == "__main__":
    pass