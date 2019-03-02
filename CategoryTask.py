# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools, random, traceback
from psychopy import visual, event
from ImageStimNumpyuByte import ImageStimNumpyuByte
from TaskControl import TaskControl
import numpy as np


class CategoryTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        self.interTrialFrames = 120
        self.maxTrialFrames = 600
        self.rewardDistance = 6 # degrees to move stim for reward
        
        # mouse can move target stimulus with wheel for early training
        # varying stimulus duration and/or masking not part of this stage
        self.moveStim = True
        self.preMoveFrames = 30 # number of frames after stimulus onset before stimulus moves
        
        # stim params
        self.stimType = 'gratings' # 'gratings' or 'image'
        self.stimSize = 10 # degrees
        self.stimFrames = [15,60] # duration of target stimulus; ignored if moveStim is True
        
        self.gratingsSF = [1,1] # cycles/deg
        self.gratingsOri = [-45,45] # clockwise degrees from vertical
        
        self.noiseSquareSize = [0.1,0.3,1,3] # degrees; temp param for testing image stim
        
        # mask params
        self.maskType = 'noise' # None, 'noise', or 'plaid'
        self.maskShape = 'target' # 'target', 'surround', 'full'
        self.maskOnset = [0,6,30] # frames >=0 relative to target stimulus onset
        self.maskFrames = 30 # duration of mask

    def checkParameterValues(self):
        pass
    
    def run(self):
        self.checkParameterValues()
        
        self.prepareRun()
        
        try:
            # create stim
            stimSizePix = int(self.stimSize*self.pixelsPerDeg)
            if self.stimType=='gratings':
                categoryParams = (self.gratingsSF,self.gratingsOri)
                stim = visual.GratingStim(win=self._win,
                                          units='pix',
                                          mask='gauss',
                                          tex='sin',
                                          size=stimSizePix, 
                                          pos=(0,0))  
            elif self.stimType=='image':
                categoryParams = (self.noiseSquareSize,)*2
                self._stimImage = np.zeros((stimSizePix,)*2,dtype=np.uint8)
                stim  = ImageStimNumpyuByte(self._win,image=self._stimImage,size=self._stimImage.shape[::-1],pos=(0,0))
            
            assert(len(categoryParams)==2)
            assert(len(categoryParams[0])==len(categoryParams[1]))
            
            # create mask
            if self.maskType=='noise':
                mask = [visual.NoiseStim(win=self._win,
                                          units='pix',
                                          mask='gauss',
                                          noiseType='Filtered',
                                          noiseFilterLower = 0/self.pixelsPerDeg,
                                          noiseFilterUpper = 10/self.pixelsPerDeg,
                                          size=stimSizePix, 
                                          pos=(0,0))]
            elif self.maskType=='plaid':
                mask = [visual.GratingStim(win=self._win,
                                           units='pix',
                                           mask='gauss',
                                           tex='sin',
                                           size=stimSizePix, 
                                           pos=(0,0),
                                           ori=ori,
                                           opacity=op)  for ori,op in zip((0,90),(1.0,0.5))]
            
            # create list of trial parameter combinations
            trialParams = [list(i) for i in itertools.product(categoryParams[0],categoryParams[1],self.stimFrames,self.maskOnset)]
            # add reward side to off diagonal elements of category parameter matrix
            # and remove combinations on diagonal
            for params in trialParams[:]: # loop through shallow copy of trialParams
                i,j = (categoryParams[n].index(params[n])+1 for n in (0,1))
                if i==j:
                    trialParams.remove(params)
                elif i/j>1:
                    params.append(1)
                else:
                    params.append(-1)
            print(trialParams)
            
            # run session
            sessionFrame = 0
            trialFrame = 0
            self.trialStartFrame = []
            self.trialRewardSide = []
            self.trialRewarded = []
            if self.stimType=='gratings':
                self.trialSF = []
                self.trialOri = []
            elif self.stimType=='image':
                self.trialSquareSize = []
            
            while True: # each loop is a frame flip
                # start new trial
                if trialFrame==0:
                    stimPos = 0
                    stim.pos = (stimPos,0)
                    trialIndex = len(self.trialStartFrame) % len(trialParams)
                    params = trialParams[trialIndex]
                    stimFrames,maskOnset,rewardSide = params[2:]
                    self.trialRewardSide.append(rewardSide)
                    self.trialStartFrame.append(sessionFrame)
                    if trialIndex==0:
                        random.shuffle(trialParams)
                    if self.stimType=='gratings':
                        sf,ori = params[:2]
                        self.trialSF.append(sf)
                        self.trialOri.append(ori)
                        stim.sf = sf/self.pixelsPerDeg
                        stim.ori = ori
                    elif self.stimType=='image':
                        squareSize = params[0]
                        self.trialSquareSize.append(squareSize)
                        self._squareSizePix = squareSize*self.pixelsPerDeg
                        self.updateStimImage(random=True)
                        stim.setReplaceImage(self._stimImage)
                    if self.maskType=='noise':
                        for m in mask:
                            m.updateNoise()
                
                # update stimulus/mask after intertrial gray screen period is complete
                if trialFrame > self.interTrialFrames:
                    if self.moveStim:
                        if trialFrame > self.interTrialFrames+self.preMoveFrames:
                            stimPos += self.translateEndoderChange()
                            stim.pos = (stimPos,0)
                            stim.draw()
                    else:
                        if trialFrame <= self.interTrialFrames+stimFrames:
                            stim.draw()
                        if self.maskType is not None and (self.interTrialFrames+maskOnset < trialFrame <= self.interTrialFrames+maskOnset+self.maskFrames):
                            for m in mask:
                                if self.maskType=='plaid' and self.stimType=='gratings':
                                    m.sf = stim.sf
                                m.draw()
                     
                self.visStimFlip()
                trialFrame += 1
                sessionFrame += 1
                
                # end trial if reward earned or at max trial duration
                if ((rewardSide<0 and stimPos<-self.rewardDistance*self.pixelsPerDeg) or
                    (rewardSide>0 and stimPos>self.rewardDistance*self.pixelsPerDeg)):
                       self.deliverReward()
                       self.trialRewarded.append(True)
                       trialFrame = 0
                elif trialFrame==self.maxTrialFrames:
                    self.trialRewarded.append(False)
                    trialFrame = 0
                
                # check for keyboard events to end session
                if len(event.getKeys()) > 0:                  
                    event.clearEvents()
                    break
                
        except:
            traceback.print_exc()
            
        finally:
            self.completeRun()
            
        
    def updateStimImage(self):
        # temp function for testing numpy array stim
        numSquares = int(round(self._stimImage.shape[0]/self._squareSizePix))
        self._stimImage = self._numpyRandom.randint(0,2,(numSquares,)*2).astype(np.uint8)*255
        self._stimImage = np.repeat(self._stimImage,self._squareSizePix,0)
        self._stimImage = np.repeat(self._stimImage,self._squareSizePix,1)


if __name__ == "__main__":
    pass