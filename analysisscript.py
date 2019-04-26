# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:05:40 2019

@author: svc_ccg
"""
import numpy as np
import h5py
import fileIO
from matplotlib import pyplot as plt
import pandas as pd

f = fileIO.getFile()
d = h5py.File(f)

trialStartFrames = d['trialStartFrame'].value
trialEndFrames = d['trialEndFrame'].value
trialRewardDirection = d['trialRewardDir'].value
trialResponse = d['trialResponse'].value
deltaWheel = d['deltaWheelPos'].value
preStimFrames = d['preStimFrames'].value
rewardFrames = d['rewardFrames'].value
openLoopFrames = d['openLoopFrames'].value

preFrames = preStimFrames + openLoopFrames
fig, ax = plt.subplots()
rightTrials = []
leftTrials = []
for i, (trialStart, trialEnd, rewardDirection, resp) in enumerate(zip(trialStartFrames, trialEndFrames, trialRewardDirection, trialResponse)):
    if i>0 and i<len(trialStartFrames):
        if abs(resp)<2:
            #get wheel position trace for this trial!
            if not 'closedLoopWheelPos' in d.keys():
                trialWheel = np.cumsum(deltaWheel[trialStart+preFrames:trialEnd])  
            else:    
                trialWheel = d['closedLoopWheelPos'][trialStart+preFrames:trialEnd]
                
                
            trialreward = np.where((rewardFrames>trialStart)&(rewardFrames<=trialEnd))[0]
            reward = rewardFrames[trialreward[0]]-trialStart-preFrames if len(trialreward)>0 else None
            if rewardDirection>0:
                ax.plot(trialWheel, 'b', alpha=0.2)
                if reward is not None:
                    ax.plot(reward, trialWheel[reward], 'bo')
                rightTrials.append(trialWheel)
            else:
                ax.plot(trialWheel, 'r', alpha=0.2)
                if reward is not None:
                    ax.plot(reward, trialWheel[reward], 'ro')
                leftTrials.append(trialWheel)
        
rightTrials = pd.DataFrame(rightTrials).fillna(np.nan).values
leftTrials = pd.DataFrame(leftTrials).fillna(np.nan).values
ax.plot(np.nanmean(rightTrials,0), 'b', linewidth=3)
ax.plot(np.nanmean(leftTrials, 0), 'r', linewidth=3)



