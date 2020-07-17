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
        edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
        target = visual.GratingStim(win=self._win,
                                    units='pix',
                                    mask=self.gratingEdge,
                                    maskParams=edgeBlurWidth,
                                    tex=self.gratingType,
                                    size=int(self.targetSize*self.pixelsPerDeg),
                                    contrast=self.targetContrast,
                                    sf=self.targetSF/self.pixelsPerDeg,
                                    ori=self.targetOri)
        
        mouse = event.Mouse(win=self._win)
        mouse.setVisible(0)
        
        self._toggle = False
        
        while self._continueSession:
            target.pos += mouse.getRel()
            
            if self._trialFrame == self.toggleInterval:
                self._trialFrame = 0
            
            if not self._toggle or self._trialFrame < self.toggleOnFrames:
                target.draw()
                
            self.showFrame()
    
    
    def showFrame(self):
        self._frameSignalOutput.write(True)
        
        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()
        
        self._sessionFrame += 1
        self._trialFrame += 1
        
        keys = event.getKeys()
        if 'escape' in keys:   
            self._continueSession = False
        elif 'space' in keys:
            self._toggle = not self._toggle
            self._trialFrame = 0
        
        self._frameSignalOutput.write(False)



if __name__ == "__main__":
    pass