# -*- coding: utf-8 -*-
"""
Created on Thu May 09 14:25:47 2019

@author: svc_ccg
"""

from __future__ import division
import os, random
import cv2 
from psychopy import visual 
from TaskControl import TaskControl


class PassiveImageChange(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        
        self.imageDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ImageSetA')
        self.imageNames = os.listdir(self.imageDir)
        self._images = [cv2.imread(os.path.join(self.imageDir,img),cv2.IMREAD_UNCHANGED).astype(float)[::-1]
                        / 255 * 2 -1 for img in self.imageNames]
        
        self.grayFrames = 60
        self.imageFrames = 30
        self.changeProb = 0.05 # probability of image change on each flash
        
        self.ledProb = 0.3 # probability that image change is triggered during image change trial
        self.ledOffsetFrames = -30 # offset between image change and led onset
        self.useLED = True
        self.ledDur = 0.004 # seconds
        self.ledRamp = 0 # seconds
        self.ledAmp = 5 # volts
        
        
    def taskFlow(self):
        
        imgInd = random.randint(0,len(self.imageNames)-1)
        imageStim = visual.ImageStim(self._win,
                                     image=self._images[imgInd],
                                     size=self._images[imgInd].shape[::-1],
                                     pos=(0,0))
        
        self.flashFrames = []
        self.changeFrames = []
        self.ledFrames = []
        self.trialImage = []
    
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # determine if image flash is change trial or not
            if self._trialFrame == 0:
                change = False if random.uniform(0,1) > self.changeProb else True
            
            # draw image
            if self._trialFrame >= self.grayFrames:
                if self._trialFrame == self.grayFrames:
                    self.flashFrames.append(self._sessionFrame)
                    if change:
                        self.changeFrames.append(self._sessionFrame)
                        imgInd = random.choice([i for i in range(len(self.imageNames)) if i != imgInd])
                        imageStim.setImage(self._images[imgInd])
                    self.trialImage.append(self.imageNames[imgInd])
                imageStim.draw()
            
            # trigger led on random change trials
            if (self._trialFrame == self.grayFrames + self.ledOffsetFrames and 
                change and random.uniform(0,1) <= self.ledProb):
                    self.ledFrames.append(self._sessionFrame)
                    self._led = True
                
            if self._trialFrame == self.grayFrames + self.imageFrames:
                self._trialFrame = -1
            
            self.showFrame()


if __name__ == "__main__":
    pass