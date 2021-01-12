# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

import itertools
from psychopy import visual  
from TaskControl import TaskControl


class RFMapping(TaskControl):
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        self.gratingCenter = []
        self.gratingContrast = [0.4,1]
        self.gratingOri = [(0,),(90,),(-45,),(45,),(0,90)] # clockwise degrees from vertical
        self.gratingSize = 25 # degrees
        self.gratingSF = 0.08 # cycles/deg
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.1 # only applies to raisedCos
        self.preFrames = 30
        self.stimFrames = [2,6]
        self.postFrames = 30
        

    def taskFlow(self):
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None

        gratings = [visual.GratingStim(win=self._win,
                                       units='pix',
                                       mask=self.gratingEdge,
                                       maskParams=edgeBlurWidth,
                                       tex=self.gratingType,
                                       size=int(self.targetSize*self.pixelsPerDeg),
                                       sf=self.targetSF/self.pixelsPerDeg,
                                       opacity=opa)
                                       for opa in (1.0,0.5)]
        
        params = itertools.product(self.gratingCenter,self.gratingContrast,self.gratingOri,self.stimFrames)
        
        trialCount = 0
        
        trialFrames = self.preFrames + max(self.stimFrames) + self.postFrames
        
        self.trialGratingCenter = []
        self.trialGratingContrast = []
        self.trialGratingOri = []
        self.trialStimFrames = []
        
        while self._continueSession:
            self.trialGratingCenter.append(params[trialCount][0])
            self.trialGratingContrast.append(params[trialCount][1])
            self.trialGratingOri.append(params[trialCount][2])
            self.trialStimFrames.append(params[trialCount][3])
            
            for g,ori in zip(gratings[:len(self.trialGratingOri[-1])],self.trialGratingOri[-1]):
                g.pos = self.trialGratingCenter[-1]
                g.contrast = self.trialGratingContrast[-1]
                g.ori = ori
                
            if self.preFrames <= self._trialFrame < self.preFrames + self.trialStimFrames[-1]:
                for _,i in enumerate(self.trialGratingOri):
                    gratings[i].draw()
                
            self.showFrame()
            
            if self._trialFrame == trialFrames:
                self._trialFrame = 0
                trialCount += 1


if __name__ == "__main__":
    pass