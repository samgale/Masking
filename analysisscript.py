# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:05:40 2019

@author: svc_ccg
"""
import numpy as np
import h5py
import fileIO
from matplotlib import pyplot as plt

f = fileIO.getFile()
d = h5py.File(f)

trialStartFrames = d['trialStartFrame'].value
trialEndFrames = d['trialEndFrame'].value
trialRewardDirection = d['trialRewardDir'].value
trialResponse = d['trialResponse'].value
deltaWheel = d['deltaWheelPos'].value
preStimFrames = d['preStimFrames'].value

preFrames = 15
fig, ax = plt.subplots()
rightTrials = []
leftTrials = []
for i, (trialStart, trialEnd, rewardDirection, resp) in enumerate(zip(trialStartFrames, trialEndFrames, trialRewardDirection, trialResponse)):
    if i>0 and i<len(trialStartFrames):
        if resp>0:
            trialWheel = np.cumsum(deltaWheel[trialStart:trialEnd])        
            if rewardDirection>0:
                ax.plot(trialWheel, 'b', alpha=0.2)
                rightTrials.append(trialWheel)
            else:
                ax.plot(trialWheel, 'r', alpha=0.2)
                leftTrials.append(trialWheel)
        
rightTrials = np.array(rightTrials)
leftTrials = np.array(leftTrials)
ax.plot(np.mean(rightTrials,0), 'b')
ax.plot(np.mean(leftTrials, 0), 'r')

#edirts