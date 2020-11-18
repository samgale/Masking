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
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        
        self.equalSampling = False # equal sampling of trial parameter combinations
        self.probGoRight = 0.5 # fraction of go trials rewarded for rightward movement of wheel
        self.probCatch = 0 # fraction of catch trials with no target and no reward
        self.probMask = 0 # fraction of trials with mask
        self.maxConsecutiveSameDir = 4
        self.maxConsecutiveMaskTrials = 4
        self.maxConsecutiveOptoTrials = 4
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 720 # max total preStim frames
        self.quiescentFrames = 60 # frames before stim onset during which wheel movement delays stim onset
        self.openLoopFramesFixed = 18 # min frames after stimulus onset before wheel movement has effects
        self.openLoopFramesVariableMean = 0 # mean of additional open loop frames drawn from exponential distribution
        self.openLoopFramesMax = 120 # max total openLoopFrames
        self.maxResponseWaitFrames = 120 # max frames between end of openLoopFrames and end of go trial
        
        self.rewardSizeLeft = self.rewardSizeRight = None # set solenoid open time in seconds; otherwise defaults to self.solenoidOpenTime
        self.wheelRewardDistance = 8.0 # mm of wheel movement to achieve reward
        self.maxQuiescentMoveDist = 1.0 # max allowed mm of wheel movement during quiescent period
        self.normRewardDistance = 0.25 # distance (normalized to screen width) target moves for reward
        self.rotateTarget = False # rotate rather than move stimulus
        self.rewardRotation = 45 # degrees target rotates for reward
        self.rewardCatchNogo = False # reward catch trials if no movement of wheel
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
        # only one of targetPos and targetOri can be len() > 1
        # parameters that can vary across trials are lists
        self.normTargetPos = [(0,0)] # normalized initial xy  position of target; center (0,0), bottom-left (-0.5,-0.5), top-right (0.5,0.5)
        self.targetFrames = [1] # duration of target stimulus
        self.targetContrast = [1]
        self.targetSize = 20 # degrees
        self.targetSF = 0.08 # cycles/deg
        self.targetOri = [-45,45] # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.1 # only applies to raisedCos
        
        # mask params
        self.maskType = None # None, 'plaid', or 'noise'
        self.normMaskPos = [[(0,0)]]
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskOnset = [15] # frames >0 relative to target stimulus onset
        self.maskFrames = [12] # duration of mask
        self.maskContrast = [1]
        
        # opto params
        self.probOpto = 0 # fraction of trials with optogenetic stimulation
        self.optoChanNames = ['left','right']
        self.optoChan = [(True,True)] # list of len(optoChanNames) lists of channels to activate
        self.optoAmp = 5 # V to led/laser driver
        self.optoOffRamp = 0.1 # duration in sec of linear off ramp
        self.optoOnset = [0] # frames >=0 relative to target stimulus onset

    
    def setDefaultParams(self,name,taskVersion=None):
        if name == 'training1':
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
            if taskVersion in ('rot','rotation'):
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
            
        elif name == 'training2':
            # learn to associate wheel movement with stimulus movement and reward
            # only use 1-2 sessions
            self.setDefaultParams('training1',taskVersion)
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
            
        elif name == 'training3':
            # introduce quiescent period, shorter response window, incorrect penalty, and catch trials
            self.setDefaultParams('training2',taskVersion)
            self.wheelRewardDistance = 3.0  # increase   
            self.quiescentFrames = 60
            self.maxResponseWaitFrames = 1200 # adjust this 
            self.useIncorrectNoise = True
            self.incorrectTimeoutFrames = 360
            self.incorrectTrialRepeats = 5 # will repeat for unanswered trials
            self.solenoidOpenTime = 0.07
            self.probCatch = 0
            
        elif name == 'training4':
            # more stringent parameters
            self.setDefaultParams('training3',taskVersion)
            self.maxResponseWaitFrames = 60
            self.incorrectTimeoutFrames = 720
            self.solenoidOpenTime = 0.05
            self.probCatch = .15
  
        elif name == 'nogo':
            self.setDefaultParams('training4',taskVersion)
            self.probCatch = 0.33
            self.rewardCatchNogo = True
            self.useGoTone = True
            
        elif name == 'training5':
            # flashed target
            self.setDefaultParams('training4',taskVersion)
            self.moveStim = False
            self.postRewardTargetFrames = 0
            self.wheelRewardDistance = 2.0
            self.targetFrames = [12] # adjust this
            
        elif name == 'testing':
            self.setDefaultParams('training5',taskVersion)
            self.equalSampling = True
            self.useIncorrectNoise = False
            self.incorrectTimeoutFrames = 0
            self.incorrectTrialRepeats = 0 
            self.targetFrames = [2]
            self.wheelRewardDistance = 2.0
            
        elif name == 'target duration':
            self.setDefaultParams('testing',taskVersion)
            self.targetFrames = [1,2,4,12]
            self.probCatch = 1 / (1 + 2*len(self.targetFrames))
        
        elif name == 'target contrast':
            self.setDefaultParams('testing',taskVersion)
            self.targetContrast = [0.2,0.4,0.6,1]
            self.probCatch = 1 / (1 + 2*len(self.targetContrast))
            
        elif name == 'masking':
            self.setDefaultParams('testing',taskVersion)
            self.maskType = 'plaid'
            self.maskShape = 'target'
            self.normMaskPos = [self.normTargetPos]
            self.maskFrames = [24]
            self.maskOnset = [2,3,4,6]
            self.maskContrast = [0.4]
            self.targetContrast = [0.4]
            self.probMask = 0.6
            self.probCatch = 1 / (1 + 2*len(self.maskOnset))
            
        elif name == 'mask pos':
            self.setDefaultParams('masking',taskVersion)
            self.normMaskPos += [[(0,-0.25),(0,0.25)]]
            self.maskOnset = [2,4,6]
            self.probCatch = 1 / (1 + 2*len(self.maskOnset))
            
        elif name == 'opto contrast':
            self.setDefaultParams('testing',taskVersion)
            self.probOpto = 0.6
            self.optoChan = [(True,True)]
            self.optoOnset = [4,6,8,10,12]
            self.targetContrast = [0.4]
            self.probCatch = 1 / (1 + 2*len(self.targetContrast))
            
        elif name == 'opto unilateral':
            self.setDefaultParams('opto contrast',taskVersion)
            self.optoChan = [(True,True),(True,False),(False,True)]
            self.optoOnset = [0]
            
        elif name == 'opto masking':
            self.setDefaultParams('masking',taskVersion)
            self.probOpto = 0.6
            self.optoChan = [(True,True)]
            self.optoOnset = [4,6,8,10,12]
            self.maskOnset = [2]
            self.probMask = 0.5
            self.probCatch = 1 / (1 + 2*len(self.maskOnset))
            
        else:
            print(str(name)+' is not a recognized set of default parameters')
    
     
    def checkParamValues(self):
        assert((len(self.normTargetPos)>1 and len(self.targetOri)==1) or
               (len(self.normTargetPos)==1 and len(self.targetOri)>1))
        assert(self.quiescentFrames <= self.preStimFramesFixed)
        assert(self.maxQuiescentMoveDist <= self.wheelRewardDistance) 
        assert(0 not in self.targetFrames + self.targetContrast + self.maskOnset + self.maskFrames + self.maskContrast)
        for prob in (self.probGoRight,self.probCatch,self.probMask):
            assert(0 <= prob <= 1)
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create target stimulus
        targetPosPix = [tuple(p[i] * self.monSizePix[i] for i in (0,1)) for p in self.normTargetPos]
        targetSizePix = int(self.targetSize * self.pixelsPerDeg)
        sf = self.targetSF / self.pixelsPerDeg
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask=self.gratingEdge,
                                    maskParams=edgeBlurWidth,
                                    tex=self.gratingType,
                                    size=targetSizePix, 
                                    sf=sf)  
        
        # create mask
        # 'target' mask overlaps target
        # 'surround' mask surrounds but does not overlap target
        # 'full' mask surrounds and overlaps target
        if self.maskShape=='target':
            maskPosPix = [[tuple(p[i] * self.monSizePix[i] for i in (0,1)) for p in m] for m in self.normMaskPos]
            maskSizePix = targetSizePix
            maskEdgeBlur = self.gratingEdge
            maskEdgeBlurWidth = edgeBlurWidth
        else:
            maskPosPix = [[(0,0)]]
            maskSizePix = max(self.monSizePix)
            maskEdgeBlur = None
            maskEdgeBlurWidth = None
        nMasks = len(maskPosPix[0])
        
        if self.maskType=='plaid':
            maskOri = (0,90) if len(self.normTargetPos)>1 else (-45,45)
            mask = [[visual.GratingStim(win=self._win,
                                        units='pix',
                                        mask=maskEdgeBlur,
                                        maskParams=maskEdgeBlurWidth,
                                        tex=self.gratingType,
                                        size=maskSizePix,
                                        sf=sf,
                                        ori=ori,
                                        opacity=opa)
                                        for ori,opa in zip(maskOri,(1.0,0.5))]
                                        for _ in range(nMasks)]
                                       
        elif self.maskType=='noise':
            maskSizePix = 2**math.ceil(math.log(maskSizePix,2))
            mask = [[visual.NoiseStim(win=self._win,
                                      units='pix',
                                      mask=maskEdgeBlur,
                                      maskParams=maskEdgeBlurWidth,
                                      noiseType='Filtered',
                                      noiseFractalPower = 0,
                                      noiseFilterOrder = 1,
                                      noiseFilterLower = 0.5*sf,
                                      noiseFilterUpper = 2*sf,
                                      size=maskSizePix)]
                                      for _ in range(nMasks)]
        
        if self.maskShape=='surround':
            mask += [[visual.Circle(win=self._win,
                                   units='pix',
                                   radius=0.5*targetSizePix,
                                   lineColor=0.5,
                                   fillColor=0.5)
                                   for pos in targetPosPix]]
        
        # define parameters for each trial type
        if len(targetPosPix) > 1:
            goRightPos = [pos for pos in targetPosPix if pos[0] < 0]
            goLeftPos = [pos for pos in targetPosPix if pos[0] > 0]
            goRightOri = goLeftOri = [self.targetOri[0]]
        else:
            goRightPos = goLeftPos = [targetPosPix[0]]
            goRightOri = [ori for ori in self.targetOri if ori < 0]
            goLeftOri = [ori for ori in self.targetOri if ori > 0]
            if not self.rotateTarget:
                goRightOri,goLeftOri = goLeftOri,goRightOri
                
        trialParams = {}
        rd = 0 if self.rewardCatchNogo else np.nan
        trialParams['catch'] = {}
        trialParams['catch']['params'] = [(rd,     # reward direction
                                          (0,0),   # target pos
                                          0,       # target ori
                                          0,       # target contrast
                                          0,       # target frames
                                          0,       # mask onset
                                          [(0,0)]*len(maskPosPix[0]), # mask pos
                                          0,       # mask frames
                                          0,       # mask contrast
                                          (False,False),  # opto chan
                                          np.nan)] # opto onset
        trialParams['catch']['count'] = 0
        
        if self.probMask > 0:
            trialParams['maskOnly'] = {}
            trialParams['maskOnly']['params'] = list(itertools.product([rd],
                                                                       [(0,0)],
                                                                       [0],
                                                                       [0],
                                                                       [0],
                                                                       [0],
                                                                       maskPosPix,
                                                                       self.maskFrames,
                                                                       self.maskContrast,
                                                                       [(False,False)],
                                                                       [np.nan]))
            trialParams['maskOnly']['count'] = 0
        
        for side,rd,pos,ori in zip(('GoLeft','GoRight'),(-1,1),(goLeftPos,goRightPos),(goLeftOri,goRightOri)):
            trialParams['targetOnly'+side] = {}
            trialParams['targetOnly'+side]['params'] = list(itertools.product([rd],
                                                                              pos,
                                                                              ori,
                                                                              self.targetContrast,
                                                                              self.targetFrames,
                                                                              [0],
                                                                              [[(0,0)]*len(maskPosPix[0])],
                                                                              [0],
                                                                              [0],
                                                                              [(False,False)],
                                                                              [np.nan]))
            trialParams['targetOnly'+side]['count'] = 0
        
        if self.probMask > 0:
            for side,rd,pos,ori in zip(('GoLeft','GoRight'),(-1,1),(goLeftPos,goRightPos),(goLeftOri,goRightOri)):
                trialParams['mask'+side] = {}
                trialParams['mask'+side]['params'] = list(itertools.product([rd],
                                                                            pos,
                                                                            ori,
                                                                            self.targetContrast,
                                                                            self.targetFrames,
                                                                            self.maskOnset,
                                                                            maskPosPix,
                                                                            self.maskFrames,
                                                                            self.maskContrast,
                                                                            [(False,False)],
                                                                            [np.nan]))
                trialParams['mask'+side]['count'] = 0
        
        if self.equalSampling and self.probGoRight==0.5:
            trialParams['targetOnly'] = {}
            trialParams['targetOnly']['params'] = trialParams['targetOnlyGoLeft']['params'] + trialParams['targetOnlyGoRight']['params']
            trialParams['targetOnly']['count'] = 0
            if self.probMask > 0:
                trialParams['mask'] = {}
                trialParams['mask']['params'] = trialParams['maskGoLeft']['params'] + trialParams['maskGoRight']['params']
                trialParams['mask']['count'] = 0
            
        if self.probOpto > 0:
            optoParams = list(itertools.product(self.optoChan,self.optoOnset))
            for trialType in list(trialParams.keys()):
                trialParams[trialType+'Opto'] = {}
                trialParams[trialType+'Opto']['params'] = [prm[:8] + op for prm in trialParams[trialType]['params'] for op in optoParams]
                trialParams[trialType+'Opto']['count'] = 0
            
        # calculate pixels to move or degrees to rotate stimulus per radian of wheel movement
        rewardMove = self.rewardRotation if self.rotateTarget else self.monSizePix[0] * self.normRewardDistance
        self.wheelGain = rewardMove / (self.wheelRewardDistance / self.wheelRadius)
        maxQuiescentMove = (self.maxQuiescentMoveDist / self.wheelRadius) * self.wheelGain
        
        # things to keep track of
        self.trialType = []
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
        self.trialMaskPos = []
        self.trialMaskFrames = []
        self.trialMaskContrast = []
        self.trialRewardDir = []
        self.trialOptoChan = []
        self.trialOptoOnset = []
        self.trialResponse = []
        self.trialResponseDir = []
        self.trialResponseFrame = []
        self.trialRepeat = [False]
        self.quiescentMoveFrames = [] # frames where quiescent period was violated
        incorrectRepeatCount = 0
        maskCount = 0
        optoCount = 0
        monitorEdge = 0.5 * (self.monSizePix[0] - targetSizePix)
        
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
                    consecutiveDir = self.trialRewardDir[-1] if len(self.trialRewardDir) >= self.maxConsecutiveSameDir and (all(d==self.trialRewardDir[-1] for d in self.trialRewardDir[-self.maxConsecutiveSameDir:]) or all(np.isnan(self.trialRewardDir[-self.maxConsecutiveSameDir:]))) else None
                    showMask = True if random.random() < self.probMask and maskCount < self.maxConsecutiveMaskTrials else False
                    maskCount = maskCount + 1 if showMask else 0
                    if random.random() < self.probCatch and consecutiveDir not in (0,np.nan):
                        trialType = 'maskOnly' if showMask else 'catch'
                    else:
                        trialType = 'mask' if showMask else 'targetOnly'
                        if not (self.equalSampling and self.probGoRight==0.5):
                            goRight = False if consecutiveDir == 1 else (True if consecutiveDir == -1 else random.random() < self.probGoRight)
                            trialType += 'GoRight' if goRight else 'GoLeft'
                    if random.random() < self.probOpto and optoCount < self.maxConsecutiveOptoTrials:
                        trialType += 'Opto'
                        optoCount += 1
                    else:
                        optoCount = 0
                    if self.equalSampling:
                        if trialParams[trialType]['count'] == len(trialParams[trialType]['params']):
                            trialParams[trialType]['count'] = 0
                        if trialParams[trialType]['count'] == 0:
                            random.shuffle(trialParams[trialType]['params'])
                        params = trialParams[trialType]['params'][trialParams[trialType]['count']]
                        trialParams[trialType]['count'] += 1
                    else:
                        params = random.choice(trialParams[trialType]['params'])
                    rewardDir,initTargetPos,initTargetOri,targetContrast,targetFrames,maskOnset,maskPos,maskFrames,maskContrast,optoChan,optoOnset = params
                    if rewardDir == 1 and self.rewardSizeRight is not None:
                        rewardSize = self.rewardSizeRight
                    elif rewardDir == -1 and self.rewardSizeLeft is not None:
                        rewardSize = self.rewardSizeLeft
                    else:
                        rewardSize = self.solenoidOpenTime
                
                targetPos = list(initTargetPos) # position of target on screen
                targetOri = initTargetOri # orientation of target on screen
                target.pos = targetPos
                target.ori = targetOri
                target.contrast = targetContrast
                target.phase = (0,0)
                if self.maskType is not None:
                    for msk,p in zip(mask,maskPos):
                        for m in msk:
                            m.pos = p
                            m.contrast = maskContrast
                            if self.maskType == 'noise' and isinstance(m,visual.NoiseStim):
                                m.updateNoise()
                self.trialType.append(trialType)
                self.trialStartFrame.append(self._sessionFrame)
                self.trialTargetPos.append(initTargetPos)
                self.trialTargetOri.append(initTargetOri)
                self.trialTargetContrast.append(targetContrast)
                self.trialTargetFrames.append(targetFrames)
                self.trialMaskOnset.append(maskOnset)
                self.trialMaskPos.append(maskPos)
                self.trialMaskFrames.append(maskFrames)
                self.trialMaskContrast.append(maskContrast)
                self.trialRewardDir.append(rewardDir)
                self.trialOptoChan.append(optoChan)
                self.trialOptoOnset.append(optoOnset)
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
                        if self.rotateTarget:
                            if self.autoRotationRate > 0:
                                deltaOri = rewardDir * self.autoRotationRate * self._win.monitorFramePeriod
                                targetOri += deltaOri
                                closedLoopWheelMove += deltaOri
                            else:
                                deltaOri = self.deltaWheelPos[-1] * self.wheelGain
                                targetOri += deltaOri
                                closedLoopWheelMove += deltaOri
                                if self.keepTargetOnScreen and abs(targetOri) > 90:
                                    adjust = targetOri - 90 if targetOri > 90 else targetOri + 90
                                    targetOri -= adjust
                                    closedLoopWheelMove -= adjust
                            target.ori = targetOri
                        else:
                            if self.normAutoMoveRate > 0:
                                deltaPos = rewardDir * self.normAutoMoveRate * self.monSizePix[0] * self._win.monitorFramePeriod
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
                            target.pos = targetPos 
                    else:
                        closedLoopWheelMove += self.deltaWheelPos[-1] * self.wheelGain
                if self.moveStim:
                    if targetFrames > 0:
                        if self.gratingDriftFreq > 0:
                            target.phase[0] += rewardDir * self.gratingDriftFreq * self._win.monitorFramePeriod
                            target.phase = target.phase
                        elif self.reversePhasePeriod > 0 and ((self._trialFrame - self.trialPreStimFrames[-1]) % self.reversePhasePeriod) == 0:
                            phase = (0.5,0) if target.phase[0] == 0 else (0,0)
                            target.phase = phase
                        target.draw()
                else:
                    if (self.maskType is not None and maskFrames > 0 and
                        (self.trialPreStimFrames[-1] + maskOnset <= self._trialFrame < 
                         self.trialPreStimFrames[-1] + maskOnset + maskFrames)):
                        for msk in mask:
                            for m in msk:
                                m.draw()
                    elif self._trialFrame < self.trialPreStimFrames[-1] + targetFrames:
                        target.draw()
                
                # turn on opto
                if self._trialFrame == self.trialPreStimFrames[-1] + optoOnset:
                    self._opto = {'ch': optoChan, 'amp': self.optoAmp, 'lastVal': self.optoAmp}
                    
                # define response if wheel moved past threshold (either side) or max trial duration reached
                # trialResponse for go trials is 1 for correct direction, -1 for incorrect direction, or 0 for no response
                # trialResponse for no-go trials is 1 for no response or -1 for movement in either direction
                # trialResponse for catch trials is nan
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
            
            # turn off opto
            if not np.isnan(optoOnset) and self._trialFrame == self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                self._opto = {'ch': optoChan, 'amp': self.optoAmp, 'offRamp': self.optoOffRamp}
                
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
                elif not np.isnan(optoOnset) and self._trialFrame < self.trialPreStimFrames[-1] + self.trialOpenLoopFrames[-1] + self.maxResponseWaitFrames:
                    pass # wait until end of response window to turn off opto
                else:
                    self.trialEndFrame.append(self._sessionFrame)
                    self._trialFrame = -1
                    if self.trialResponse[-1] < 1 and incorrectRepeatCount < self.incorrectTrialRepeats:
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