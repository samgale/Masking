# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:05:40 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import h5py
import fileIO
from matplotlib import pyplot as plt
import pandas as pd 
import probeData

f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
d = h5py.File(f)

trialStartFrames = d['trialStartFrame'].value
trialEndFrames = d['trialEndFrame'].value
trialRewardDirection = d['trialRewardDir'].value
trialResponse = d['trialResponse'].value
deltaWheel = d['deltaWheelPos'].value
preStimFrames = d['preStimFrames'].value
openLoopFrames = d['openLoopFrames'].value

if 'rewardFrames' in d.keys():
    rewardFrames = d['rewardFrames'].value
else:
    if 'responseFrames' in d.keys():
        responseTrials = np.where(trialResponse!= 0)[0]
        rewardFrames = d['responseFrames'].value[trialResponse[responseTrials]>0]
    else:
        rewardFrames = d['trialResponseFrame'].value[trialResponse>0]
        

preFrames = preStimFrames + openLoopFrames
fig, ax = plt.subplots()

# for rightTrials stim presented on L, turn right - viceversa for leftTrials
rightTrials = []
leftTrials = []
for i, (trialStart, trialEnd, rewardDirection, resp) in enumerate(zip(trialStartFrames, trialEndFrames, trialRewardDirection, trialResponse)):
    if i>0 and i<len(trialStartFrames):
        if abs(resp)<2:
            #get wheel position trace for this trial!
            if True: #not 'closedLoopWheelPos' in d.keys():
                trialWheel = np.cumsum(deltaWheel[trialStart+preFrames:trialEnd])  
            else:    
                trialWheel = d['closedLoopWheelPos'][trialStart+preFrames:trialEnd]
                
                
            trialreward = np.where((rewardFrames>trialStart)&(rewardFrames<=trialEnd))[0]
            reward = rewardFrames[trialreward[0]]-trialStart-preFrames if len(trialreward)>0 else None
            if rewardDirection>0:
                ax.plot(trialWheel, 'r', alpha=0.2)
                if reward is not None:
                    ax.plot(reward, trialWheel[reward], 'ro')
                rightTrials.append(trialWheel)
            else:
                ax.plot(trialWheel, 'b', alpha=0.2)
                if reward is not None:
                    ax.plot(reward, trialWheel[reward], 'bo')
                leftTrials.append(trialWheel)
        
rightTrials = pd.DataFrame(rightTrials).fillna(np.nan).values
leftTrials = pd.DataFrame(leftTrials).fillna(np.nan).values
ax.plot(np.nanmean(rightTrials,0), 'r', linewidth=3)
ax.plot(np.nanmean(leftTrials, 0), 'b', linewidth=3)

probeData.formatFigure(fig, ax, xLabel='Frame Number', yLabel='Wheel Position (pix)')

trials = np.count_nonzero(trialResponse)

print('\n' + 'repeat incorrect trials: '+str(d['repeatIncorrectTrials'].value))

percentResponse = trials/len(trialResponse)
print('percent response was: {}'.format(percentResponse))

correct = (trialResponse==1).sum()
percentCorrect = correct/float(trials)
print('percent correct was: {}'.format(percentCorrect))


