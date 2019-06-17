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
import scipy.stats


f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
d = h5py.File(f)

trialRewardDirection = d['trialRewardDir'].value[:-1]
trialResponse = d['trialResponse'].value
#targetLengths = d['targetFrames'].value
targetLengths = d['maskOnset'][:]
#targetLengths[0]=30


#trialTargetFrames = d['trialTargetFrames'][:-1]
trialTargetFrames = d['trialMaskOnset'][:-1]
trialTargetFrames[np.isnan(trialTargetFrames)] = 30 

# [R stim] , [L stim]
hits = [[],[]]
misses = [[], []]
noResps = [[],[]]
for i, direction in enumerate([-1,1]):
    directionResponses = [trialResponse[(trialRewardDirection==direction) & (trialTargetFrames == tf)] for tf in np.unique(trialTargetFrames)]
    hits[i].append([np.sum(drs==1) for drs in directionResponses])
    misses[i].append([np.sum(drs==-1) for drs in directionResponses])
    noResps[i].append([np.sum(drs==0) for drs in directionResponses])

hits = np.squeeze(np.array(hits))
misses = np.squeeze(np.array(misses))
noResps = np.squeeze(np.array(noResps))
totalTrials = hits+misses+noResps


for num, denom, title in zip([hits, hits, hits+misses], 
                             [totalTrials, hits+misses, totalTrials],
                             ['Total hit rate', 'Response hit rate', 'Total response rate']):
    fig, ax = plt.subplots()
    ax.plot(np.unique(trialTargetFrames), num[0]/denom[0], 'ro-')
    ax.plot(np.unique(trialTargetFrames), num[1]/denom[1], 'bo-')
    '''chanceRates = [[[i/n for i in scipy.stats.binom.interval(0.95,n,0.5)] for n in h] for h in denom]
    chanceRates = np.array(chanceRates)
    for val, chanceR, chanceL in zip(np.unique(targetLengths), chanceRates[0], chanceRates[1]):
       plt.plot([val, val], chanceR, color='red', alpha=.5)     # 0 and 1 = R and L, respectively
       plt.plot([val+.2, val+.2], chanceL, color='blue', alpha=.5)'''
    formatFigure(fig, ax, xLabel='SOA (frames)', yLabel='percent trials', 
                 title=title + " :  " + '-'.join(f.split('_')[-3:-1]))
    ax.set_xlim([-2, trialTargetFrames[0]+2])
    ax.set_ylim([0,1.01])
    ax.set_xticks(np.unique(trialTargetFrames))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
            
    if 30 in trialTargetFrames:   
        a = ax.get_xticks().tolist()
        a = [int(i) for i in a]    
        a[-1]='no mask' 
        ax.set_xticklabels(a)



