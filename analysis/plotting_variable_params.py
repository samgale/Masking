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
from ignoreTrials import ignore_trials
from dataAnalysis import ignore_after, get_dates, import_data


def plot_param(data, param='targetLength', showTrialN=True, 
               ignoreNoRespAfter=None, returnCounts=False, array_only=False):
    '''
    plots the percent correct or response for a variable target duration session,
    variable contrast session, opto contrast session, or masking session
    
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
    trialType = d['trialType'][:end]


# determine parameter to analyze    
    if param =='targetLength' or param=='target duration':
        trialParam = d['trialTargetFrames'][:end] * 1000/framerate 
    elif param =='targetContrast':
        trialParam = d['trialTargetContrast'][:end]
    elif param =='opto':
        trialParam = d['trialOptoOnset'][:end]
    elif param == 'soa' or param=='masking':
        param='soa'
        trialParam = d['trialMaskOnset'][:end]
    
    
# if there are repeats, identify and ignore them
    if 'trialRepeat' in d.keys():
        prevTrialIncorrect = d['trialRepeat'][:end]  #recommended, since keeps track of how many repeats occurred 
    else:
        if d['incorrectTrialRepeats'][()] > 0:
            prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))
        else:
            prevTrialIncorrect = np.full(end, False)
            


# remove ignore trials
    ignore = ignore_trials(d) 
    ignoring = np.full(end, 0)
    
    for i,_ in enumerate(ignoring):
        if i in ignore:
            ignoring[i] = 1
            
    trialResponse2 = trialResponse[ignoring==0]                    
    trialParam = trialParam[ignoring==0]
    trialRewardDirection = trialRewardDirection[ignoring==0]   
    prevTrialIncorrect = prevTrialIncorrect[ignoring==0]
    trialType = trialType[ignoring==0]
    trialTargetContrast = d['trialTargetContrast'][:end][ignoring==0]

# ignore repeats AND catch trials (no target)
    trialResponse2 = trialResponse2[(prevTrialIncorrect==False) & (trialType!='catch')]                    
    trialParam = trialParam[(prevTrialIncorrect==False) & (trialType!='catch')]
    trialRewardDir = trialRewardDirection[(prevTrialIncorrect==False) & (trialType!='catch')]    
    trialTargetContrast = trialTargetContrast[(prevTrialIncorrect==False) & (trialType!='catch')]


# for session with opotgenetics, selects only those trials with the optogenetics
    if param=='opto':
        optoOnset = d['optoOnset'][:]
        for i, trial in enumerate(trialParam):  # replace nans (no opto) with -1
            if ~np.isfinite(trial):
                noOpto = optoOnset[-1] + np.median(np.round(np.diff(optoOnset))) if len(optoOnset)>1 else optoOnset[0]+1
                trialParam[i] = noOpto


# now that repeats and catch have been removed, identify parameter values
    paramVals = np.unique(trialParam)


# ignore param values that == 0, these are no target    
    if param == 'targetLength' or param == 'targetContrast':
        paramVals = paramVals[paramVals>0]


#handling mask only vs no mask (both onset == 0)    
    if param == 'soa':  
        paramVals = paramVals[paramVals>0]
        noMaskVal = paramVals[-1] + round(np.mean(np.diff(paramVals)))  # assigns noMask condition an evenly-spaced value from soas
        paramVals = np.append(paramVals, noMaskVal)              # makes final value the no-mask condition
        
        trialMaskFrames = d['trialMaskFrames'][:end][ignoring==0][(prevTrialIncorrect==False) & (trialType!='catch')]    
        trialResponseDir = d['trialResponseDir'][:end][ignoring==0][(prevTrialIncorrect==False) & (trialType!='catch')] 
        trialMaskContrast = d['trialMaskContrast'][:end][ignoring==0][(prevTrialIncorrect==False) & (trialType!='catch')] 
        
    # filters target-Only trials
        for i, (mask, frames) in enumerate(zip(trialParam, trialMaskFrames)):    
            if frames==0 and mask==0:
                trialParam[i]=noMaskVal
                
                #what about mask-only condition? -- trialParam of 0
 

    
# separate trials into [[turn l] , [turn R]] and parameter value
     
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    
    for i, direction in enumerate([-1,1]):
        directionResponses = [trialResponse2[(trialRewardDir==direction) & 
                                             (trialParam == tf)] for tf in paramVals]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    resps = hits+misses
    totalTrials = resps+noResps
    
    
    if param=='opto':
        trialType = d['trialType'][:end][ignoring==0]
        trialTurn = d['trialResponseDir'][:end][ignoring==0]
        optoOnset = d['trialOptoOnset'][:end][ignoring==0]
        
        for i, trial in enumerate(trialTurn):
            if ~np.isfinite(trial):
                trialTurn[i] = 0    
                
        for i, trial in enumerate(optoOnset):  # replace nans (no opto) with -1
            if ~np.isfinite(trial):
                optoOnset[i]= noOpto
           
        catch = [[] for i in range(len(paramVals))]    # separate catch trials by opto onset
        
        for i, o in enumerate(paramVals):
            for j, (opto, turn, typ) in enumerate(zip(optoOnset, trialTurn, trialType)):
               if 'catch' in typ:
                    if o==opto:
                        catch[i].append(abs(int(turn)))
    
                            
        catchTurn = np.array(list(map(sum, [c for c in catch])))
        catchCounts = np.array(list(map(len, [c for c in catch])))
        
        xlabls1 = [(on/framerate) - ((1/framerate)*2) for on in list(paramVals)]
        xticklabels = [int(np.round(x*1000)) for x in xlabls1]
        
        
    if param=='soa':
        maskOnlyTotal = np.sum(trialType=='maskOnly')  # ignores are already excluded
        maskOnly = [[], [], []]
        for typ, resp, mask in zip(trialType, trialResponseDir, trialMaskContrast):
            if typ == 'maskOnly' and mask>0:
                if np.isfinite(resp):
                    if resp==1:
                        maskOnly[0].append(resp)   # turned right 
                    elif resp==-1:
                        maskOnly[1].append(resp)  # turned left
                else:
                    maskOnly[2].append(1)  # no response
                    
        maskOnly[0] = np.sum(maskOnly[0])  # turn R 
        maskOnly[1] = np.sum(maskOnly[1])*-1  # turn L
        maskOnly[2] = np.sum(maskOnly[2])  # no resp
    
    
    
    if 0 in trialRewardDir:        # this already excludes repeats 
    
        nogoTotal = len(trialParam[trialParam==0])
        nogoCorrect = len(trialResponse2[(trialResponse2==1) & (trialParam==0)])
        nogoMove = nogoTotal - nogoCorrect
        
        nogoTurnDir = np.array(nogo_turn(d)[0][0])
        
        nogoR = sum(nogoTurnDir[nogoTurnDir==1])
        nogoL = sum(nogoTurnDir[nogoTurnDir==-1])*-1
    else:
        pass
    

    
    if returnCounts==True:
        array_counts = {str(param): list(paramVals), 'total trials': totalTrials, 
                        'hits': hits, 'misses': misses, 'resps': resps, 'no response': noResps}
        return (mouse, array_counts)
    
    elif array_only==True:
        
        mask =  (maskOnly[0]+maskOnly[1])/maskOnlyTotal if param=='soa' else None
        avg_catch = (catchTurn/catchCounts) if param=='opto' else None
        
        return (mouse, {str(param):np.round(paramVals, 2), 
                        'Response Rate':(resps[0]+resps[1])/(totalTrials[0]+totalTrials[1]), 
                        'Fraction Correct':(hits[0]+hits[1])/(resps[0]+resps[1]), 
                        'Catch Trials':avg_catch,
                        'maskOnly': mask})
    

## PLOTTING   
    else:   
        for num, denom, title in zip([hits, hits, resps], 
                                     [totalTrials,resps, totalTrials],
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
                xticklabels = [(np.round((x/framerate)*1000, 2)) for x in xticklabels]
                
            elif param=='targetContrast':
                xlab = 'Target Contrast'
                
            elif param=='opto':
                
                if title == 'Response Rate':
                    ax.plot(xticks, catchTurn/catchCounts, 'mo-', alpha=.5, lw=3, label='Catch Trials') # plot catch trials
                    for p, c in zip(xticks, catchCounts):
                        fig.text(p, 1.05, str(c), transform=ax.transData, color='m', alpha=.5, fontsize=10,ha='center',va='bottom')
                
                
                
                xticklabels[-1] = 'no opto'
                ax.xaxis.set_label_coords(0.5,-0.08)  
                xlab = 'Optogenetic light onset relative to target onset (ms)'
                
                
            elif param=='soa':
                
                xticklabels = [int(np.round((tick/framerate)*1000)) for tick in xticklabels]
                
                xlab = 'Mask Onset From Target Onset (ms)'
                lbl = ['mask\nonly', 'target only']
                del xticklabels[-1]
                xticklabels.append(lbl[1])
                if title=='Response Rate':  # show mask-only resps 
                    xticks = np.insert(xticks, 0, 1)
                    xticklabels = np.insert(xticklabels, 0, lbl[0])
                    ax.plot(1, maskOnly[0]/maskOnlyTotal, 'ro')
                    ax.plot(1, maskOnly[1]/maskOnlyTotal, 'bo')
                    fig.text(1,1.05,str(maskOnlyTotal),transform=ax.transData,color='k',fontsize=10,ha='center',va='bottom')
                ax.xaxis.set_label_coords(0.5,-0.08)

            
            if title=='Response Rate':   #no go
                if 0 in trialRewardDir:
                    ax.plot(0, nogoCorrect/nogoTotal, 'ko', ms=8)
                    ax.plot(0, nogoR/nogoMove, 'r>', ms=8)  #plot the side that was turned in no-go with an arrow in that direction
                    ax.plot(0, nogoL/nogoMove, 'b<', ms=8)
                    xticks = np.concatenate(([0],xticks))
                    xticklabels = ['no go'] + xticklabels
                                
            if showTrialN==True:
                for x,Ltrials,Rtrials in zip(paramVals,denom[0], denom[1]):   #deom[0]==L, denom[1]==R
                    for y,n,clr in zip((1.1,1.15),[Rtrials, Ltrials],'rb'):
                        fig.text(x,y,str(n),transform=ax.transData,color=clr,fontsize=10,ha='center',va='bottom')
        

            formatFigure(fig, ax, xLabel=xlab, yLabel=title)
            fig.suptitle(('(' + mouse + '    ' + date + ')'), fontsize=13)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
            if param=='targetLength':
                ax.set_xlim([-5, paramVals[-1]+1])
            elif param=='targetContrast':
                ax.set_xlim([0, 1.05])
            elif param=='soa':
                ax.set_xlim([0, max(paramVals)+1])
            elif param=='opto':
                ax.set_xlim([paramVals[0]-1, max(paramVals)+1])
            else:
                ax.set_xlim([-.5, max(paramVals)+1])
                
            ax.set_ylim([0,1.05])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.subplots_adjust(top=0.84, bottom=0.105, left=0.095, right=0.92, hspace=0.2, wspace=0.2)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
            
