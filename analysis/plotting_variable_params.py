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


def plot_param(data, param='targetLength', showTrialN=True, ignoreNoRespAfter=None, returnArray=False):
    '''
    plots the percent correct or response for a variable target duration session
    
    param is variable parameter that you want to visualize - targetLength, targetContrast, opto, soa
    showTrialN=True will add the total counts into the plots 
    ignoreNoResp takes an int, and will ignore all trials after [int] consecutive no resps
    returnArray=True is for the save plot, will return just the values and no plots
    '''
    
    d = data
    matplotlib.rcParams['pdf.fonttype'] = 42
    
# get file data
    info = str(d).split('_')[-3:-1]
    date = get_dates(info[1])
    mouse = info[0]
   

# if sequence of no resps exceeds val assigned to 'ignoreNoRespAfter' in function call,
# returns trial num to ignore trials after 
    end = ignore_after(d, ignoreNoRespAfter)[0] if ignoreNoRespAfter is not None else len(d['trialResponse'][:])
    
    
# assign relevant file values to variables 
    trialResponse = d['trialResponse'][:end]
    trialRewardDirection = d['trialRewardDir'][:end]   
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))


# determine parameter to analyze    
    if param =='targetLength':
        trialParam = d['trialTargetFrames'][:end] * 1000/framerate 
    elif param =='targetContrast':
        trialParam = d['trialTargetContrast'][:end]
    elif param =='opto':
        trialParam = d['trialOptoOnset'][:end]
    elif param == 'soa':
        trialParam = d['trialMaskOnset'][:end]
    
    
# if there are repeats, identify and ignore them
    if 'trialRepeat' in d.keys():
        prevTrialIncorrect = d['trialRepeat'][:end]  #recommended, since keeps track of how many repeats occurred 
    else:
        prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))


# ignore repeats AND catch trials (no target)   
    trialResponse2 = trialResponse[(prevTrialIncorrect==False) & (np.isfinite(trialRewardDirection))]                    
    trialParam = trialParam[(prevTrialIncorrect==False) & (np.isfinite(trialRewardDirection))]
    trialRewardDirection = trialRewardDirection[(prevTrialIncorrect==False) & (np.isfinite(trialRewardDirection))]      


# for session with opotgenetics, selects only those trials with the optogenetics
    if param=='opto':
        for i, trial in enumerate(trialParam):  # replace nans (no opto) with -1
            if ~np.isfinite(trial):
                trialParam[i] = -1


# now that repeats and catch have been removed, identify parameter values
    paramVals = np.unique(trialParam)


# ignore param values that == 0, these are no target    
    if param == 'targetLength' or param == 'targetContrast':
        paramVals = paramVals[paramVals>0]


#handling mask only vs no mask (both onset == 0)    
    if param == 'soa':   
        noMaskVal = paramVals[-1] + round(np.mean(np.diff(paramVals)))  # assigns noMask condition an evenly-spaced value from soas
        paramVals = np.append(paramVals, noMaskVal)              # makes final value the no-mask condition
        
    # filters target-Only trials
        for i, (mask, trial) in enumerate(zip(trialParam, 
               d['trialTargetFrames'][:end][(prevTrialIncorrect==False) & (np.isfinite(d['trialRewardDir'][:]))])):    
            if trial>0 and mask==0:
                trialParam[i]=noMaskVal
 

    
# separate trials into [[turn l] , [turn R]] and parameter value
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    
    for i, direction in enumerate([-1,1]):
        directionResponses = [trialResponse2[(trialRewardDirection==direction) & 
                                             (trialParam == tf)] for tf in paramVals]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    totalTrials = hits+misses+noResps
    
    
    if 0 in trialRewardDirection:        # this already excludes repeats 
    
        nogoTotal = len(trialParam[trialParam==0])
        nogoCorrect = len(trialResponse2[(trialResponse2==1) & (trialParam==0)])
        nogoMove = nogoTotal - nogoCorrect
        
        nogoTurnDir = np.array(nogo_turn(d)[0][0])
        
        nogoR = sum(nogoTurnDir[nogoTurnDir==1])
        nogoL = sum(nogoTurnDir[nogoTurnDir==-1])*-1
    else:
        pass
    
    
    if returnArray==True:
        array_counts = {str(param): paramVals, 'total trials': totalTrials, 
                        'hits': hits, 'misses': misses, 'no response': noResps}
        return array_counts
    
    
    elif returnArray==False:   
        for num, denom, title in zip([hits, hits, hits+misses], 
                                     [totalTrials, hits+misses, totalTrials],
                                     ['Fraction Correct', 'Fraction Correct Given Response', 'Response Rate']):
            
            
            fig, ax = plt.subplots()
            ax.plot(paramVals, num[0]/denom[0], 'bo-', lw=3, alpha=.7, label='Left turning')  #here [0] is right trials and [1] is left
            ax.plot(paramVals, num[1]/denom[1], 'ro-', lw=3, alpha=.7, label='Right turning')
            ax.plot(paramVals, (num[0]+num[1])/(denom[0]+denom[1]), 'ko--', alpha=.5, label='Combined average')  #plots the combined average 
           

                        
            xticks = paramVals
            xticklabels = list(paramVals)

            if param=='targetLength':
                xticklabels = list(np.round(xticks).astype(int))
                xlab = 'Target Duration (ms)'
            elif param=='targetContrast':
                xlab = 'Target Contrast'
            elif param=='opto':
                xlab = 'Opto Onset'
                x,lbl = ([0],['no\nopto'])
                xticks = np.concatenate((x,xticks))
                xticklabels = lbl+xticklabels
                ax.xaxis.set_label_coords(0.5,-0.08)
                
            elif param=='soa':
                xlab = 'Mask Onset From Target Onset (ms)'
                if title=='Response Rate':
                    x,lbl = ([0],['mask\nonly'])
                    xticks = np.concatenate((x,xticks))
                    xticklabels = lbl+xticklabels
                    ax.xaxis.set_label_coords(0.5,-0.08)
                
            if title=='Response Rate':   #no go
                if 0 in trialRewardDirection:
                    ax.plot(0, nogoCorrect/nogoTotal, 'ko', ms=8)
                    ax.plot(0, nogoR/nogoMove, 'r>', ms=8)  #plot the side that was turned in no-go with an arrow in that direction
                    ax.plot(0, nogoL/nogoMove, 'b<', ms=8)
                    xticks = np.concatenate(([0],xticks))
                    xticklabels = ['no go']+xticklabels
             
                                
            if showTrialN==True:
                for x,Rtrials,Ltrials in zip(paramVals,denom[0], denom[1]):
                    for y,n,clr in zip((1.05,1.1),[Rtrials, Ltrials],'rb'):
                        fig.text(x,y,str(n),transform=ax.transData,color=clr,fontsize=10,ha='center',va='bottom')
        

            formatFigure(fig, ax, xLabel=xlab, yLabel=title)
            fig.suptitle(('(' + mouse + '    ' + date + ')'), fontsize=13)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
            if param=='targetLength':
                ax.set_xlim([-5, paramVals[-1]+1])
            elif param=='targetContrast':
                ax.set_xlim([0, 1.05])
            else:
                ax.set_xlim([-.5, max(paramVals)+1])
                
            ax.set_ylim([0,1.05])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.subplots_adjust(top=0.86, bottom=0.105, left=0.095, right=0.92, hspace=0.2, wspace=0.2)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
            
            
#            
#            for x,y,z in zip(trialResponse2, d['trialOptoOnset'][:len(trialResponse2)], trialRewardDirection):
#                if np.isfinite(y):
#                    print('finite:  ', x,y,z)
#                else:
#                    print('no finite:  ', x,y,z)
#                        
