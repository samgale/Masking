# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:05:13 2019

@author: svc_ccg
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure
from dataAnalysis import get_dates, import_data

      """ After you get that working, the next thing is to take the combined right/left curve 
       for each optoOnset and put them all on the same plot. 
       Try black for no opto (nan) and different shades of blue for each of the non-nan onsets. 
       There will be four lines total for the params I'm am planning today. 
       Do this for both of the plots: fraction of trials with response and fraction correct given response.  """   


def plot_opto_vs_param(data, param = 'targetContrast'):
    
    '''
    plotting optogenetic onset to other variable parameter in session
    param == 'targetContrast', 'targetLength', 'soa'
    '''
    
    matplotlib.rcParams['pdf.fonttype'] = 42

    d = data
    info = str(d).split('_')[-3:-1]
    date = get_dates(info[1])
    mouse = info[0]
    
    
    trialResponse = d['trialResponse'][:]
    end = len(trialResponse)
    
    trialRewardDirection = d['trialRewardDir'][:end]
    optoOnset = d['optoOnset'][:]
    trialOptoOnset = d['trialOptoOnset'][:end]
    
    for i, trial in enumerate(trialOptoOnset):  # replace nans (no opto) with -1
        if ~np.isfinite(trial):
            trialOptoOnset[i] = -1

# select paramter to evaluate 
    if param == 'targetContrast':
        trialParam = d['trialTargetContrast'][:end]
    elif param == 'targetLength':
        trialParam = d['trialTargetFrames'][:end]
    elif param == 'soa':
        trialParam = d['trialMaskOnset'][:end]
        

# list of unique paramter values, depending on what param was declared            
    paramVals = np.unique(trialParam)    
    param_num = len(np.unique(paramVals))
    
    if param == 'targetFrames' or param == 'targetContrast':
        paramVals = paramVals[paramVals>0]
    
# remove catch trials 
    trialOptoOnset = trialOptoOnset[(np.isfinite(trialRewardDirection)==True)]
    trialResponse = trialResponse[(np.isfinite(trialRewardDirection)==True)]
    trialParam = trialParam[(np.isfinite(trialRewardDirection)==True)]
    trialRewardDirection = trialRewardDirection[(np.isfinite(trialRewardDirection)==True)]
    
# lists of responses by param for each opto level 
    hitsR = [[] for i in range(param_num)]
    missesR = [[] for i in range(param_num)]
    noRespsR = [[] for i in range(param_num)]
     
    for i, val in enumerate(np.unique(paramVals)):
        responsesR = [trialResponse[(trialParam==val) & (trialRewardDirection==1) & 
                                       (trialOptoOnset == op)] for op in np.unique(trialOptoOnset)]
        hitsR[i].append([np.sum(drs==1) for drs in responsesR])
        missesR[i].append([np.sum(drs==-1) for drs in responsesR])
        noRespsR[i].append([np.sum(drs==0) for drs in responsesR])
    
    hitsR = np.squeeze(np.array(hitsR))
    missesR = np.squeeze(np.array(missesR))
    noRespsR = np.squeeze(np.array(noRespsR))
    totalTrialsR = hitsR+missesR+noRespsR
    
    
    hitsL = [[] for i in range(param_num)]
    missesL = [[] for i in range(param_num)]
    noRespsL = [[] for i in range(param_num)]
    
    
    for i, val in enumerate(np.unique(paramVals)):     
        responsesL = [trialResponse[(trialParam==val) & (trialRewardDirection==-1) & 
                                        (trialOptoOnset == op)] for op in np.unique(trialOptoOnset)]
        hitsL[i].append([np.sum(drs==1) for drs in responsesL])
        missesL[i].append([np.sum(drs==-1) for drs in responsesL])
        noRespsL[i].append([np.sum(drs==0) for drs in responsesL])
            
    hitsL = np.squeeze(np.array(hitsL))
    missesL = np.squeeze(np.array(missesL))
    noRespsL = np.squeeze(np.array(noRespsL))
    totalTrialsL = hitsL+missesL+noRespsL
    
    averages = []
    for i, value in enumerate(paramVals):
        for Rnum, Lnum, Rdenom, Ldenom, title in zip([hitsR, hitsR+missesR], 
                                                     [hitsL, hitsL+missesL],
                                                     [hitsR+missesR, totalTrialsR],
                                                     [hitsR+missesL, totalTrialsL],
                                                     ['Fraction Correct Given Response', 'Response Rate']):
                    
                    fig, ax = plt.subplots()
                    ax.plot(optoOnset, Rnum[i]/Rdenom[i], 'bo-', lw=3, alpha=.7, label='Right turning')  
                    ax.plot(optoOnset, Lnum[i]/Ldenom[i], 'ro-', lw=3, alpha=.7, label='Left turning')
                    ax.plot(optoOnset, (Rnum[i]+Lnum[i])/(Rdenom[i]+Ldenom[i]), 'ko--', alpha=.5, 
                            label='Combined average')  
                    averages.append((Rnum[i]+Lnum[i])/(Rdenom[i]+Ldenom[i]))
                   
                    xticks = optoOnset
                    xticklabels = list(optoOnset)
        
                    showTrialN=True           
                    if showTrialN==True:
                        for x,Rtrials,Ltrials in zip(optoOnset,Rdenom[i], Ldenom[i]):
                            for y,n,clr in zip((1.05,1.1),[Rtrials, Ltrials],'rb'):
                                fig.text(x,y,str(n),transform=ax.transData,color=clr,fontsize=10,
                                         ha='center',va='bottom')
                
                    if param == 'contrast':
                        pval = 'Target Contrast'
                    elif param == 'targetLength':
                        pval = 'Target Length'
                    elif param == 'soa':
                        pval = 'SOA'
                        
                    xlab = 'Opto onset from target onset'
                    
                    formatFigure(fig, ax, xLabel=xlab, yLabel=title)
                    
                    if value >0:
                        fig.suptitle(('(' + mouse + ',   ' + date + ')    ' + 
                                  pval + ' = ' + str(value), fontsize=13)
                    else
                                    
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
    
                    ax.set_xlim([-1, max(optoOnset)+1])  
                    ax.set_ylim([0,1.05])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False)
                    plt.subplots_adjust(top=0.86, bottom=0.105, left=0.095, right=0.92, hspace=0.2, wspace=0.2)
                    plt.legend(loc='best', fontsize='small', numpoints=1) 
                    
                    
                
                