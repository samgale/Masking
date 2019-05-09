# -*- coding: utf-8 -*-
"""
Created on Thu May 09 14:25:47 2019

@author: svc_ccg
"""

from __future__ import division
import os, random
import cv2 
from ImageStimNumpyuByte import ImageStimNumpyuByte  
from TaskControl import TaskControl


class PassiveImageChange(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        
        self.imageDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ImageSetA')
        self.imageNames = os.listdir(self.imageDir)
        self.images = [cv2.imread(os.path.join(self.imageDir,img),cv2.IMREAD_UNCHANGED) for img in self.imageNames]
        
        self.grayFrames = 60 # 0.5 seconds for 120 Hz monitor
        self.imageFrames = 30
        
        self.changeProb = 0.05 # probability of image change for each flash
        self.rewardProb = 0.3 # probability that image change is rewarded
        self.rewardOffsetFrames = -30 # offset between image change and reward delivery
        self.solenoidOpenTime = 0.004 # duration in seconds of reward ouptut signal
        
        
    def taskFlow(self):
        
        imgInd = random.randint(0,len(self.images)-1)
        imageStim = ImageStimNumpyuByte(self._win,
                                        image=self.images[imgInd],
                                        size=self.images[imgInd].shape[::-1],
                                        pos=(0,0))
        
        self.stimFrames = []
        self.changeFrames = []
        self.rewardFrames = []
    
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # determine if image flash is change trial or not
            if self._trialFrame == 0:
                change = False if random.uniform(0,1) > self.changeProb else True
            
            # draw image
            if self._trialFrame >= self.grayFrames:
                if self._trialFrame == self.grayFrames:
                    self.stimFrames.append(self._sessionFrame)
                    if change:
                        self.changeFrames.append(self._sessionFrame)
                        imgInd = random.choice([i for i in range(len(self.images)) if i != imgInd])
                        imageStim.setReplaceImage(self.images[imgInd])
                imageStim.draw()
            
            # deliver reward on random change trials
            if (change and self._trialFrame == self.grayFrames + self.rewardOffsetFrames and 
                random.uniform(0,1) <= self.rewardProb):
                self._reward = True
            
            self.showFrame()


if __name__ == "__main__":
    pass