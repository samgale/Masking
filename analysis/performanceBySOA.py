# -*- coding: utf-8 -*-
"""
Created on Wed Jun 05 14:08:12 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import h5py, os
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure
import fileIO


f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
d = h5py.File(f)

trialRewardDirection = d['trialRewardDir'].value[:-1]
trialResponse = d['trialResponse'].value
#targetLengths = d['targetFrames'].value
trialLengths = d['maskOnset'][:]
trialLengths[0]=-1

#trialTargetFrames = d['trialTargetFrames'][:-1]
trialTargetFrames = d['trialMaskOnset'][:-1]
trialTargetFrames[np.isnan(trialTargetFrames)] = -1


hits = [[],[]]
misses = [[], []]
noResps = [[],[]]
for i, direction in enumerate([-1,1]):
    directionResponses = [trialResponse[(trialRewardDirection==direction) & (trialTargetFrames == tf)] for tf in np.unique(targetLengths)]
    hits[i].append([np.sum(drs==1) for drs in directionResponses])
    misses[i].append([np.sum(drs==-1) for drs in directionResponses])
    noResps[i].append([np.sum(drs==0) for drs in directionResponses])

hits = np.squeeze(np.array(hits))
misses = np.squeeze(np.array(misses))
noResps = np.squeeze(np.array(noResps))
totalTrials = hits+misses+noResps

for num, denom, title in zip([hits, hits, noResps], [totalTrials, hits+misses, totalTrials], ['total hit rate', 'response hit rate', 'no response rate']):
    fig, ax = plt.subplots()
    ax.plot(np.unique(targetLengths), num[0]/denom[0], 'ro-')
    ax.plot(np.unique(targetLengths), num[1]/denom[1], 'bo-')
    ax.set_ylim([0,1.01])
    ax.set_xlim([0, targetLengths[-1]*1.1])
    ax.set_xticks(np.unique(targetLengths))
    formatFigure(fig, ax, xLabel='SOA (frames)', yLabel='percent trials', title=title)

