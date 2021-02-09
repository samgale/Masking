# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

import itertools, random
import numpy as np
from psychopy import visual  
from TaskControl import TaskControl


class RFMapping(TaskControl):
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        self.defaultParams = 'mask mapping'
        self.gratingCenter = [(0,0)]
        self.gratingContrast = [1]
        self.gratingOri = [(0,90)] # clockwise degrees from vertical
        self.gratingSize = 25 # degrees
        self.gratingSF = 0.08 # cycles/deg
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.1 # only applies to raisedCos
        self.preFrames = 50
        self.stimFrames = [6]
        self.postFrames = 50      


    def taskFlow(self):
        
        if 'mask' in self.defaultParams:
            r = 0.5*self.gratingSize*self.pixelsPerDeg
            w,h = self.monSizePix
            self.gratingCenter = []
            for x in (r,0.25*w,0.5*w-r):
                for y in (0.5*h-r,0,-0.5*h+r):
                    self.gratingCenter.append((x,y))
            if self.defaultParams == 'masking':
                self.gratingContrast = [0.4,1]
                self.gratingOri = [(0,np.nan),(90,np.nan),(-45,np.nan),(45,np.nan),(0,90)]
                self.gratingSize = 25
                self.gratingSF = 0.08
                self.gratingType = 'sqr'
                self.gratingEdge= 'raisedCos'
                self.gratingEdgeBlurWidth = 0.1
                self.stimFrames = [2,24]
        
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None

        gratings = [visual.GratingStim(win=self._win,
                                       units='pix',
                                       mask=self.gratingEdge,
                                       maskParams=edgeBlurWidth,
                                       tex=self.gratingType,
                                       size=int(self.gratingSize*self.pixelsPerDeg),
                                       sf=self.gratingSF/self.pixelsPerDeg,
                                       opacity=opa)
                                       for opa in (1.0,0.5)]
        
        params = list(itertools.product(self.gratingCenter,self.gratingContrast,self.gratingOri,self.stimFrames))
        random.shuffle(params)
        
        self.stimStartFrame = []
        self.trialGratingCenter = []
        self.trialGratingContrast = []
        self.trialGratingOri = []
        self.trialStimFrames = []
        
        trialIndex = 0
        loopCount = 0
        
        while self._continueSession:
            if self._trialFrame == 0:
                self.trialGratingCenter.append(params[trialIndex][0])
                self.trialGratingContrast.append(params[trialIndex][1])
                self.trialGratingOri.append(params[trialIndex][2])
                self.trialStimFrames.append(params[trialIndex][3])
                
                for g,ori in zip(gratings,self.trialGratingOri[-1]):
                    if not np.isnan(ori):
                        g.pos = self.trialGratingCenter[-1]
                        g.contrast = self.trialGratingContrast[-1]
                        g.ori = ori
                
            if self.preFrames <= self._trialFrame < self.preFrames + self.trialStimFrames[-1]:
                if self._trialFrame == self.preFrame:
                    self.stimStartFrame.append(self._sessionFrame)
                for g,ori in zip(gratings,self.trialGratingOri[-1]):
                    if not np.isnan(ori):
                        g.draw()
                
            self.showFrame()
            
            if self._trialFrame == self.preFrames + self.trialStimFrames[-1] + self.postFrames:
                self._trialFrame = 0
                trialIndex += 1
                if trialIndex == len(params):
                    loopCount += 1
                    print('completed loop '+str(loopCount))
                    trialIndex = 0
                    random.shuffle(params)


if __name__ == "__main__":
    pass