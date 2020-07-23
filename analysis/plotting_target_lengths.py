# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 18:07:23 2019

@author: svc_ccg
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure
from nogoData import nogo_turn
from dataAnalysis import ignore_after, get_dates


def plot_flash(data,showTrialN=False, ignoreNoResp=None, returnArray=False):
    
    matplotlib.rcParams['pdf.fonttype'] = 42

    d = data
    info = str(d).split('_')[-3:-1]
    date = get_dates(info[1])
    mouse = info[0]
    trialResponse = d['trialResponse'][:]
    
    end = ignore_after(d, 10)[0] if ignoreNoResp is not None else len(trialResponse)
    
    trialResponse = trialResponse[:end]
    trialRewardDirection = d['trialRewardDir'][:end]    # leave off last trial, ended session before answer 
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))
    trialTargetFrames = d['trialTargetFrames'][:end] * 1000/framerate  
    repeats = d['incorrectTrialRepeats'][()]   # max num of repeats 
    
    if 'trialRepeat' in d.keys():
        prevTrialIncorrect = d['trialRepeat'][:end]  #recommended, since keeps track of how many repeats occurred 
    else:
        prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))         # array of boolean values about whethe the trial before was incorr
    trialResponse2 = trialResponse[(prevTrialIncorrect==False)]                    # false = not a repeat, true = repeat
    trialRewardDirection = trialRewardDirection[prevTrialIncorrect==False]      # use this to filter out repeated trials 
    trialTargetFrames = trialTargetFrames[prevTrialIncorrect==False]
    
    targetFrames = np.unique(trialTargetFrames)
    targetFrames = targetFrames[targetFrames>0]
    
    # [R stim] , [L stim]
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    
    for i, direction in enumerate([-1,1]):
        directionResponses = [trialResponse2[(trialRewardDirection==direction) & (trialTargetFrames == tf)] for tf in targetFrames]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    totalTrials = hits+misses+noResps
    
    
    if 0 in trialRewardDirection:        # this already excludes repeats 
    
        nogoTotal = len(trialTargetFrames[trialTargetFrames==0])
        nogoCorrect = len(trialResponse2[(trialResponse2==1) & (trialTargetFrames==0)])
        nogoMove = nogoTotal - nogoCorrect
        
        nogoTurnDir = np.array(nogo_turn(d)[0][0])
        
        nogoR = sum(nogoTurnDir[nogoTurnDir==1])
        nogoL = sum(nogoTurnDir[nogoTurnDir==-1])*-1
    else:
        pass
    
    
    if returnArray==True:
        array_counts = {'target frames': targetFrames, 'total trials': totalTrials, 
                        'hits': hits, 'misses': misses, 'no response': noResps}
        return array_counts
        
    elif returnArray==False:   
        for num, denom, title in zip([hits, hits, hits+misses], 
                                     [totalTrials, hits+misses, totalTrials],
                                     ['Fraction Correct', 'Fraction Correct Given Response', 'Response Rate']):
            fig, ax = plt.subplots()
            ax.plot(targetFrames, num[0]/denom[0], 'bo-', lw=3, alpha=.7, label='Right turning')  #here [0] is right trials and [1] is left
            ax.plot(targetFrames, num[1]/denom[1], 'ro-', lw=3, alpha=.7, label='Left turning')
            ax.plot(targetFrames, (num[0]+num[1])/(denom[0]+denom[1]), 'ko--', alpha=.5, label='Combined average')  #plots the combined average 
            y=(num[0]/denom[0])
            y2=(num[1]/denom[1])
            if showTrialN:
                for i, length in enumerate(targetFrames):
                    plt.annotate(str(denom[0][i]), xy=(length,y[i]), xytext=(5, -10), textcoords='offset points')  #adds total num of trials
                    plt.annotate(str(denom[1][i]), xy=(length,y2[i]), xytext=(5, -10), textcoords='offset points')
            
            xticks = targetFrames
            xticklabels = list(np.round(xticks).astype(int))
            if title=='Response Rate':
                if 0 in trialRewardDirection:
                    ax.plot(0, nogoCorrect/nogoTotal, 'ko', ms=8)
                    ax.plot(0, nogoR/nogoMove, 'r>', ms=8)  #plot the side that was turned in no-go with an arrow in that direction
                    ax.plot(0, nogoL/nogoMove, 'b<', ms=8)
                    xticks = np.concatenate(([0],xticks))
                    xticklabels = ['no go']+xticklabels
            
            tar = np.append(targetFrames, targetFrames)
            
            for x,Rtrials,Ltrials in zip(tar,denom[0], denom[1]):
                for y,n,clr in zip((1.05,1.1),[Rtrials, Ltrials],'rb'):
                    fig.text(x,y,str(n),transform=ax.transData,color=clr,fontsize=10,ha='center',va='bottom')
    

            formatFigure(fig, ax, xLabel='Target Duration (ms)', yLabel=title)
            fig.suptitle((mouse + '    ' + date), fontsize=12)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim([-5, targetFrames[-1]+1])
            ax.set_ylim([0,1.05])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.subplots_adjust(top=0.875, bottom=0.105, left=0.095, right=0.955, hspace=0.2, wspace=0.2)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
            
            
            
            