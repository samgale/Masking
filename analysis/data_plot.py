# -*- coding: utf-8 -*-
"""
Created on Wed May 01 14:50:27 2019

@author: svc_ccg
"""

"""
This plots how many choices were correct (excluding the incorrect repeats) compared to chance,
compared across all mice.  Returns a subplot of all 5 mice percentages over time, chance is shaded region in green.
Prints mouse IDs to console when data is compiled

"""

from __future__ import division
import os
import numpy as np
import h5py
import datetime 
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl
from behaviorAnalysis import formatFigure
mpl.rcParams['pdf.fonttype']=42

#import pandas as pd

# get hdf5 files for each mouse
def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'training_to_analyze')
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]     

# calculate how many trials they responded to, and of those what percent were correct    
def trials(data):
    trialResponse = d['trialResponse'].value
    trials = np.count_nonzero(trialResponse)
    correct = (trialResponse==1).sum()
    percentCorrect =( correct/float(trials)) * 100
    return percentCorrect


#mice = ['439508', '439502', '441357', '441358']
mice = ['460313', '460312', '460314', '457228']

fig, axes = plt.subplots(len(mice),1, facecolor='white')
if len(mice)==1:
    axes = [axes]

for im, (ax,mouse) in enumerate(zip(axes, mice)):
    files = get_files(mouse)
    print(mouse)
    hitRate = []
    chanceRates = []
    for i,f in enumerate(files):
        d = h5py.File(f)
        resp = d['trialResponse'][:]
        postHitTrials = np.concatenate(([False],resp[:-1]==1))   #trials after an incorrect repeat
        if 'normIncorrectDistance' in d:
            dist = d['normIncorrectDistance'].value*d['monSizePix'].value[0]
            preFrames = d['preStimFrames'].value+d['openLoopFrames'].value
            deltaWheel = d['deltaWheelPos']
            trialRewDir = d['trialRewardDir']
            for trial,(start,end) in enumerate(zip(d['trialStartFrame'][:],d['trialEndFrame'][:])):
                wheel = np.cumsum(deltaWheel[start+preFrames:end])          #change in wheel each trial
                rewardDir  = trialRewDir[trial]                             # -1 for turn L (R stim), 1 for turn R (L stim)
                if (rewardDir>0 and np.any(wheel>dist)) or(rewardDir<0 and np.any(wheel<-dist)): 
                    resp[trial] = 1
        resp = resp[postHitTrials] if 'incorrectTrialRepeats' in d and d['incorrectTrialRepeats'].value else resp
        hitRate.append(np.sum(resp==1)/np.sum(resp!=0))
        chanceRates.append(np.array(scipy.stats.binom.interval(0.95, np.sum(resp!=0), 0.5))/np.sum(resp!=0))
        #print(postHitTrials.sum(),trials(d),np.sum(resp==1)/np.sum(resp!=0))
        d.close()
    
    chanceRates = np.array(chanceRates)
    axes[im].text(0.1, 0.9, mouse)
    axes[im].set_ylim([0,1])
    axes[im].fill_between(np.arange(len(files)), chanceRates[:, 0], chanceRates[:, 1], color='g', alpha=0.2)
    axes[im].plot(hitRate, 'ko-')
    axes[im].set_yticks(np.arange(0, 1.01, step=0.5))
    if im<len(mice)-1:
        axes[im].tick_params(labelbottom='off')
    else:
        axes[im].set_xlabel('Session number')
        axes[im].set_ylabel('Hit Rate')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
 

     


