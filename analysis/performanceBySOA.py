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


def plot_soa(data):
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    d=data
    trialResponse = d['trialResponse'][:]
    trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]
    maskOnset = d['maskOnset'][()]                  
    trialMaskOnset = d['trialMaskOnset'][:len(trialResponse)]
    trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]       
    trialMaskContrast = d['trialMaskContrast'][:len(trialResponse)]     
    framerate = int(round(d['frameRate'][()]))
    
    noMaskVal = maskOnset[-1] + round(np.mean(np.diff(maskOnset)))  # assigns noMask condition an evenly-spaced value from soas
    maskOnset = np.append(maskOnset, noMaskVal)              # makes final value the no-mask condition
        
    for i, (mask, trial) in enumerate(zip(trialMaskOnset, trialTargetFrames)):   # filters target-Only trials 
        if trial>0 and mask==0:
            trialMaskOnset[i]=noMaskVal
            
    trialMaskOnset = np.ceil(trialMaskOnset * (1000/framerate))
    maskOnset = np.ceil(maskOnset * (1000/framerate))
    
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
    
    
    ## Mask-only responses 
    maskTotal = len(trialResponse[(trialMaskContrast>0)])
    maskOnlyTotal = len(trialResponse[(trialMaskContrast>0) & (trialTargetFrames==0)])   # rotation task 'mask only' trials can't be 'correct'
    maskOnlyCorr = len(trialResponse[(trialMaskContrast>0) & (trialResponse==1) & (trialTargetFrames==0)])
    
    stimStart = d['trialStimStartFrame'][:len(trialResponse)]
    trialOpenLoop = d['trialOpenLoopFrames'][:len(trialResponse)]
    trialRespFrames = d['trialResponseFrame'][:len(trialResponse)]
    deltaWheel = d['deltaWheelPos'][:]
    
    maskStimStart = stimStart[(trialTargetFrames==0) & (trialMaskContrast>0)]             
    maskTrialRespFrames = trialRespFrames[(trialTargetFrames==0) & (trialMaskContrast>0)]
    
    startWheelPos = []
    endWheelPos = []
    
    # we want to see which direction they moved the wheel on mask-only trials 
    for i, (start, end) in enumerate(zip(maskStimStart, maskTrialRespFrames)):    #maskOnly
        endWheelPos.append(deltaWheel[end])
        startWheelPos.append(deltaWheel[start])
    
    maskEnd = np.array(endWheelPos)
    maskStart = np.array(startWheelPos)
    maskWheelPos = maskEnd - maskStart
    
    maskOnlyTurnDir = []
    
    for j in maskWheelPos:
        if j>0:
            maskOnlyTurnDir.append(1)
        else:
            maskOnlyTurnDir.append(-1)
     
    maskOnlyTurnDir = np.array(maskOnlyTurnDir)
    maskOnlyR = sum(maskOnlyTurnDir==1)
    maskOnlyL = sum(maskOnlyTurnDir==-1)   
    
    ## no go trial responses
    trialMaskContrast= d['trialMaskContrast'][:len(trialResponse)]
    nogoResp = trialResponse[(trialTargetFrames==0) & (trialMaskContrast==0)]
    
    nogoStimStart = stimStart[(trialTargetFrames==0) & (trialMaskContrast==0)]             
    nogoTrialRespFrames = trialRespFrames[(trialTargetFrames==0) & (trialMaskContrast==0)]
    
    nogoStartWheelPos = []
    nogoEndWheelPos = []
    
    # we want to see which direction they moved the wheel on an incorrect no-go
    for (start, end, resp) in zip(nogoStimStart, nogoTrialRespFrames, nogoResp):   
        if resp==-1:
            nogoEndWheelPos.append(deltaWheel[end])
            nogoStartWheelPos.append(deltaWheel[start])
        
    nogoEndWheelPos = np.array(nogoEndWheelPos)
    nogoStartWheelPos = np.array(nogoStartWheelPos)   
    wheelPos = nogoEndWheelPos - nogoStartWheelPos
    
    nogoTurnDir = []
    
    for i in wheelPos:
        if i >0:
            nogoTurnDir.append(1)
        else:
            nogoTurnDir.append(-1)
    
    nogoTurnDir = np.array(nogoTurnDir)
    
    nogoR = sum(nogoTurnDir[nogoTurnDir==1])
    nogoL = sum(nogoTurnDir[nogoTurnDir==-1])*-1
     
    nogoTotal = len(nogoResp)
    nogoCorrect = len(trialResponse[(trialResponse==1) & (trialTargetFrames==0) & (trialMaskContrast==0)])
    nogoMove = len(nogoTurnDir) 
    nogoTurnDir = np.array(nogoTurnDir)
    
    
    for num, denom, title in zip(
            [hits, hits, respOnly],
            [totalTrials, respOnly, totalTrials],
            ['Percent Correct', 'Percent Correct Given Response', 'Total Response Rate']):
        
        fig, ax = plt.subplots()
    
        ax.plot(np.unique(maskOnset), num[0]/denom[0], 'ro-', lw=3, alpha=.7)  #here [0] is right turning trials and [1] is left turning
        ax.plot(np.unique(maskOnset), num[1]/denom[1], 'bo-', lw=3, alpha=.7)
        y=(num[0]/denom[0])
        y2=(num[1]/denom[1])
        for i, length in enumerate(np.unique(maskOnset)):
            plt.annotate(str(denom[0][i]), xy=(length,y[i]), xytext=(0, 10), textcoords='offset points')  #adds total num of trials
            plt.annotate(str(denom[1][i]), xy=(length,y2[i]), xytext=(0, -20), textcoords='offset points')
        ax.plot(np.unique(maskOnset), (num[0]+num[1])/(denom[0]+denom[1]), 'ko--', alpha=.5)  #plots the combined average  
        if title!='Percent Correct' and 0 in trialTargetFrames:
            ax.plot(0, (maskOnlyR/maskOnlyTotal), 'r>', ms=8)   #plot the side that was turned in no-go with an arrow in that direction
            ax.plot(0, (maskOnlyL/maskOnlyTotal), 'b<', ms=8)
            ax.plot(0, ((maskOnlyTotal-maskOnlyCorr)/maskOnlyTotal), 'ko')
            ax.annotate(str(maskOnlyTotal), xy=(1, (maskOnlyTotal-maskOnlyCorr)/maskOnlyTotal), xytext=(0,0), textcoords='offset points')
            ax.annotate(str(maskOnlyR), xy=(1,maskOnlyR/maskOnlyTotal), xytext=(0, 0), textcoords='offset points')
            ax.annotate(str(maskOnlyL), xy=(1,maskOnlyL/maskOnlyTotal), xytext=(0, 0), textcoords='offset points')
              
            ax.plot(-15, nogoMove/nogoTotal, 'go')
            ax.plot(-15, nogoR/nogoMove, 'r>', ms=8)   #plot the side that was turned in no-go with an arrow in that direction
            ax.plot(-15, nogoL/nogoMove, 'b<', ms=8)
            ax.annotate(str(nogoMove), xy=(-14, nogoMove/nogoTotal), xytext=(0,0), textcoords='offset points')
            ax.annotate(str(nogoR), xy=(-14,nogoR/nogoMove), xytext=(0, 0), textcoords='offset points')
            ax.annotate(str(nogoL), xy=(-14,nogoL/nogoMove), xytext=(0, 0), textcoords='offset points')
            
        formatFigure(fig, ax, xLabel='SOA (ms)', yLabel='Percent Trials', 
                     title=title + " :  " + '-'.join(str(d).split('_')[-3:-1]))
        ax.set_xlim([-25, maskOnset[-1]+10])
        ax.set_ylim([0,1.1])
        xticks = np.insert(maskOnset, 0, [-15,0])
        ax.set_xticks(xticks[2:-1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
                
      
        if title!='Percent Correct':
            ax.set_xticks(xticks)
            a = ax.get_xticks().tolist()
            a = [int(i) for i in a]     
            a[-1]='no \n mask' 
            if maskOnlyTotal:
                a[0]='no \ngo'
                a[1]='mask \n only'
            ax.set_xticklabels(a)
             
        plt.tight_layout() 
        
    plt.show()