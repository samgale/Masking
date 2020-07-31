# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:59:31 2019

@author: chelsea.strawder


This is for plotting masking sessions in the rotation mice, where there are no nogos
and we want to see their performance plotted against 'no mask' trials
Plots the percent correct (either no mvmt for nogos or not moving on maskOnly).  Rarely do they
withhold mvmt on the maskOnly trials, so this gives us a window into their bias

"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure
from nogoData import nogo_turn


def plot_soa(data,showTrialN=True,showNogo=True):
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    d=data
    trialResponse = d['trialResponse'][:]
    trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]
    trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]       
    trialMaskContrast = d['trialMaskContrast'][:len(trialResponse)]     
    framerate = 120#round(d['frameRate'][()])
    maskOnset = d['maskOnset'][()] * 1000/framerate              
    trialMaskOnset = d['trialMaskOnset'][:len(trialResponse)] * 1000/framerate
    
    noMaskVal = maskOnset[-1] + round(np.mean(np.diff(maskOnset)))  # assigns noMask condition an evenly-spaced value from soas
    maskOnset = np.append(maskOnset, noMaskVal)              # makes final value the no-mask condition
        
    for i, (mask, trial) in enumerate(zip(trialMaskOnset, trialTargetFrames)):   # filters target-Only trials 
        if trial>0 and mask==0:
            trialMaskOnset[i]=noMaskVal
    
    # [turn R] , [turn L]
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    
    for i, direction in enumerate([1,-1]):
        directionResponses = [trialResponse[(trialRewardDirection==direction) & (trialMaskOnset==soa)] for soa in np.unique(maskOnset)]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    totalTrials = hits+misses+noResps
    respOnly = hits+misses
    
    turns, ind = nogo_turn(d, returnArray=True)  # returns arrays with turning direction as 1/-1
    nogoTurnTrial = ind[0]  # list of indices of trials where turning occured
    maskTurnTrial = ind[1]
     
    nogoTotal = len(trialResponse[(trialTargetFrames==0) & (trialMaskContrast==0)])
    #nogoCorrect = len(trialResponse[(trialResponse==1) & (trialTargetFrames==0) & (trialMaskContrast==0)])  sanity check
    nogoMove = len(nogoTurnTrial) 
    nogoR = turns[0].count(1)
    nogoL = turns[0].count(-1)
    
    #maskTotal = len(trialResponse[(trialMaskContrast>0)])  sanity check
    maskOnlyTotal = len(trialResponse[(trialMaskContrast>0) & (trialTargetFrames==0)])   # rotation task 'mask only' trials can't be 'correct'
    maskOnlyCorr = len(trialResponse[(trialMaskContrast>0) & (trialResponse==1) & (trialTargetFrames==0)])
    maskOnlyR = turns[1].count(1)
    maskOnlyL = turns[1].count(-1) 
        
    for num, denom, title in zip(
            [hits, hits, respOnly],
            [totalTrials, respOnly, totalTrials],
            ['Fraction Correct', 'Fraction Correct Given Response', 'Response Rate']):
        
        fig, ax = plt.subplots()
    
        ax.plot(maskOnset, num[0]/denom[0], 'ro-', lw=3, alpha=.7)  #here [0] is right turning trials and [1] is left turning
        ax.plot(maskOnset, num[1]/denom[1], 'bo-', lw=3, alpha=.7)

        if title=='Response Rate':
            ax.plot(0, (maskOnlyR/maskOnlyTotal), 'r>', ms=8)   #plot the side that was turned in no-go with an arrow in that direction
            ax.plot(0, (maskOnlyL/maskOnlyTotal), 'b<', ms=8)
            ax.plot(0, ((maskOnlyTotal-maskOnlyCorr)/maskOnlyTotal), 'ko')
            if showTrialN:
                ax.annotate(str(maskOnlyTotal), xy=(1, (maskOnlyTotal-maskOnlyCorr)/maskOnlyTotal), xytext=(0,0), textcoords='offset points')
                ax.annotate(str(maskOnlyR), xy=(1,maskOnlyR/maskOnlyTotal), xytext=(0, 0), textcoords='offset points')
                ax.annotate(str(maskOnlyL), xy=(1,maskOnlyL/maskOnlyTotal), xytext=(0, 0), textcoords='offset points')
             
            if showNogo:
                if len(turns[0])>0:
                    ax.plot(-15, nogoMove/nogoTotal, 'ko')
                    ax.plot(-15, nogoR/nogoMove, 'r>', ms=8)   #plot the side that was turned in no-go with an arrow in that direction
                    ax.plot(-15, nogoL/nogoMove, 'b<', ms=8)
                    if showTrialN:
                        ax.annotate(str(nogoMove), xy=(-14, nogoMove/nogoTotal), xytext=(0,0), textcoords='offset points')
                        ax.annotate(str(nogoR), xy=(-14,nogoR/nogoMove), xytext=(0, 0), textcoords='offset points')
                        ax.annotate(str(nogoL), xy=(-14,nogoL/nogoMove), xytext=(0, 0), textcoords='offset points')
                else:
                    ax.plot(-15, 0, 'go')
        formatFigure(fig, ax, xLabel='Stimulus Onset Asynchrony (ms)', yLabel=title, 
                     title=str(d).split('_')[-3:-1])
        
        xticks = maskOnset
        xticklabels = list(np.round(xticks).astype(int))
        xticklabels[-1] = 'no mask'
        if title=='Response Rate':
            x,lbl = ([-15,0],['no\ngo','mask\nonly']) if showNogo else ([0],['mask\nonly'])
            xticks = np.concatenate((x,xticks))
            xticklabels = lbl+xticklabels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim([-20,xticks[-1]+10])
        ax.set_ylim([0,1.1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        if title=='Response Rate':
            ax.xaxis.set_label_coords(0.5,-0.08)
             
