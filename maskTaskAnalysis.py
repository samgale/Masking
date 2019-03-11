# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

from __future__ import division
import fileIO
import h5py
import numpy as np
import matplotlib.pyplot as plt


f = fileIO.getFile()

d = h5py.File(f)

frameRate = int(round(d['frameRate'].value))

response = d['trialResponse'][:]

endFrame = d['trialEndFrame'][:]

maskOnset = d['trialMaskOnset'][:response.size]



maskOnset[np.isnan(maskOnset)] = 2*np.nanmax(maskOnset)
maskOnsets = np.unique(maskOnset)
incorrect = np.zeros(maskOnsets.size)
noResp = incorrect.copy()
correct = incorrect.copy()
fracCorrect = incorrect.copy()
reactionTime = incorrect.copy()
for i,mo in enumerate(maskOnsets):
    incorrect[i],noResp[i],correct[i] = [np.sum(response[maskOnset==mo]==j) for j in (-1,0,1)]
    fracCorrect[i] = (correct[i]/(correct[i]+incorrect[i]))
    reactionTime[i] = endFrame[maskOnset==mo].mean()/frameRate
    
    
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot([0,maskOnsets[-1]],[0.5,0.5],'--',color='0.5')
ax.plot(maskOnsets/frameRate,fracCorrect,'ko',ms=10)
for side in ('top','right'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(np.array([-0.02,1.02])*maskOnsets[-1]/frameRate)
ax.set_ylim([0,1.02])
ax.set_xlabel('Stimulus onset asynchrony (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot(maskOnsets/frameRate,reactionTime,'ko',ms=10)
for side in ('top','right'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(np.array([-0.02,1.02])*maskOnsets[-1]/frameRate)
#ax.set_ylim([0,1.02])
ax.set_xlabel('Stimulus onset asynchrony (ms)',fontsize=16)
ax.set_ylabel('Response time (ms)',fontsize=16)
plt.tight_layout()




