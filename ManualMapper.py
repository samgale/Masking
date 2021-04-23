# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from psychopy import visual, event   
from TaskControl import TaskControl


class ManualMapper(TaskControl):
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        self.saveParams = False
        self.targetType = 'plaid' # 'grating' or 'plaid'
        self.targetContrast = 1
        self.targetSize = 25 # degrees
        self.targetSF = 0.08 # cycles/deg
        self.targetOri = 0 # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.1 # only applies to raisedCos
        self.toggleOnFrames = 24
        self.toggleInterval = 240
        

    def taskFlow(self):
        orientation = [self.targetOri]
        opacity = [1.0]
        if self.targetType == 'plaid':
            orientation.append(self.targetOri+90)
            opacity.append(0.5)
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
        target = [visual.GratingStim(win=self._win,
                                     units='pix',
                                     mask=self.gratingEdge,
                                     maskParams=edgeBlurWidth,
                                     tex=self.gratingType,
                                     size=int(self.targetSize*self.pixelsPerDeg),
                                     contrast=self.targetContrast,
                                     sf=self.targetSF/self.pixelsPerDeg,
                                     ori=ori,
                                     opacity=opa)
                                     for ori,opa in zip(orientation,opacity)]
        
        self._mouse.setVisible(0)
        
        self._toggle = False
        
        while self._continueSession:
            pos = self._mouse.getRel()
            for s in target:
                s.pos += pos
            
            if 't' in self._keys:
                self._toggle = not self._toggle
                self._trialFrame = 0
            elif self.toggle and self._trialFrame == self.toggleInterval:
                self._trialFrame = 0
            
            if not self._toggle or self._trialFrame < self.toggleOnFrames:
                for s in target:
                    s.draw()
                
            self.showFrame()


if __name__ == "__main__":
    pass