# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:57:58 2019

@author: chelsea.strawder
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure
from nogoData import nogo_turn

def plot_contrast(data,showTrialN=True):
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    d=data
    trialResponse = d['trialResponse'][:]
    trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]    # leave off last trial, ended session before answer
    trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]
    trialTargetContrast = d['trialTargetContrast'][:len(trialResponse)]
    targetContrast = d['targetContrast'][:]
    repeats = d['incorrectTrialRepeats'][()]
    
    
    if 'trialRepeat' in d.keys():
        prevTrialIncorrect = d['trialRepeat'][:len(trialResponse)]  #recommended, since keeps track of how many repeats occurred 
    else:
        prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))     # array of boolean values, if trial before was incorr
    trialResponse2 = trialResponse[prevTrialIncorrect==False]                   # false = not a repeat, true = repeat
    trialRewardDirection = trialRewardDirection[prevTrialIncorrect==False]      # use this to filter out repeated trials 
    trialTargetContrast = trialTargetContrast[prevTrialIncorrect==False]
    trialTargetFrames = trialTargetFrames[prevTrialIncorrect==False]
    
    
    # [R stim] , [L stim]
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    
    for i, direction in enumerate([-1,1]):
        directionResponses = [trialResponse2[
                (trialRewardDirection==direction) & (trialTargetContrast == tc)] for tc in np.unique(targetContrast)]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    totalTrials = hits+misses+noResps
    
    # here call no_go movement function? 
    
    if 0 in trialTargetFrames:        # this already excludes repeats 
    
        nogoTurn, _, ind = nogo_turn(d, returnArray=True)  # returns arrays with turning direction as 1/-1
        nogoTurnTrial = ind[0]  # list of indices of trials where turning occured
         
        nogoTotal = len(trialResponse[(trialTargetFrames==0) & (trialMaskContrast==0)])
        #nogoCorrect = len(trialResponse[(trialResponse==1) & (trialTargetFrames==0) & (trialMaskContrast==0)])  sanity check
        nogoMove = len(nogoTurnTrial) 
        nogoR = sum(nogoTurn==1)
        nogoL = sum(nogoTurn==-1)
               
    #misses = np.insert(misses, 0, [no_goR, no_goL], axis=1)  #add the no_go move trials to misses array 
    
    
    for num, denom, title in zip([hits, hits, hits+misses],
                                 [totalTrials, hits+misses, totalTrials],
                                 ['Fraction Correct', 'Fraction Correct Given Response', 'Response Rate']):
        fig, ax = plt.subplots()
        ax.plot(np.unique(targetContrast), num[0]/denom[0], 'bo-', lw=3, alpha=.7)  #here [0] is right trials and [1] is left
        ax.plot(np.unique(targetContrast), num[1]/denom[1], 'ro-', lw=3, alpha=.7)
       #ax.plot(np.unique(targetContrast), (num[0]+num[1])/(denom[0]+denom[1]), 'ko--', alpha=.3)  #plots the combined average 
        y1=(num[0]/denom[0])
        y2=(num[1]/denom[1])
        if showTrialN:
            for i, length in enumerate(np.unique(targetContrast)):
                plt.annotate(str(denom[0][i]), xy=(length,y1[i]), xytext=(5, -10), textcoords='offset points')  
                plt.annotate(str(denom[1][i]), xy=(length,y2[i]), xytext=(-10, 10), textcoords='offset points')
        
        xticks = targetContrast
        xticklabels = list(xticks)
        if title=='Response Rate' and 0 in trialTargetFrames:
            ax.plot(0, nogoMove/nogoTotal, 'ko', ms=8) 
            ax.plot(0, nogoR/nogoMove, 'r>', ms=8)  #plot the side that was turned in no-go with an arrow in that direction
            ax.plot(0, nogoL/nogoMove, 'b<', ms=8)  #add counts
            xticks = np.concatenate(([0],xticks))
            xticklabels = ['no go']+xticklabels
           
        formatFigure(fig, ax, xLabel='Target Contrast', yLabel=title, 
                     title=str(d).split('_')[-3:-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim([-.1, targetContrast[-1]+.1])
        ax.set_ylim([0,1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
            
    plt.show()