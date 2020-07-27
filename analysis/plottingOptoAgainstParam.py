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

""" 
After you get that working, the next thing is to take the combined right/left curve 
for each optoOnset and put them all on the same plot. 
Try black for no opto (nan) and different shades of blue for each of the non-nan onsets. 
There will be four lines total for the params I am planning today. 
Do this for both of the plots: fraction of trials with response and fraction correct given response.  
"""   


def plot_opto_vs_param(data, param = 'targetContrast', plotType = 'optoByParam' ):
    
    '''
    plotting optogenetic onset to other variable parameter in session
    param == 'targetContrast', 'targetLength', 'soa'
    plotType == 'optoByParam' or 'combined_opto'
    former creates 2x np.unique(paramVals) where opto is x axis and L and R are plotted 
    latter creates 2 plots, each with x axis as paramVal, and average L & R performance for opto as plots
    '''
    d = data
    matplotlib.rcParams['pdf.fonttype'] = 42

   
    info = str(d).split('_')[-3:-1]
    date = get_dates(info[1])
    mouse = info[0]
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))
    
    
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
    opto_num = len(np.unique(trialOptoOnset))
    optoOnset = np.unique(trialOptoOnset)
    
    if param == 'targetLength' or param == 'targetContrast':
        paramVals = paramVals[paramVals>0]
        
    param_num = len(np.unique(paramVals))

    
# remove catch trials 
    trialOptoOnset = trialOptoOnset[(np.isfinite(trialRewardDirection)==True)]
    trialResponse = trialResponse[(np.isfinite(trialRewardDirection)==True)]
    trialParam = trialParam[(np.isfinite(trialRewardDirection)==True)]
    trialRewardDirection = trialRewardDirection[(np.isfinite(trialRewardDirection)==True)]
    
    

    # each list is for an opto onset, and the values within are by paramValue  
    #  ex:  list1 = opto Onset 0, within are trials for each contrast level [0, .2, .4, 1]    
    # trials where mouse should turn Right for reward
    hitsR = [[] for i in range(opto_num)]
    missesR = [[] for i in range(opto_num)]
    noRespsR = [[] for i in range(opto_num)]
     
    for i, op in enumerate(np.unique(trialOptoOnset)):
        responsesR = [trialResponse[(trialOptoOnset==op) & (trialRewardDirection==1) & 
                                       (trialParam==val)] for val in paramVals]
        hitsR[i].append([np.sum(drs==1) for drs in responsesR])
        missesR[i].append([np.sum(drs==-1) for drs in responsesR])
        noRespsR[i].append([np.sum(drs==0) for drs in responsesR])
    
    hitsR = np.squeeze(np.array(hitsR))
    missesR = np.squeeze(np.array(missesR))
    noRespsR = np.squeeze(np.array(noRespsR))
    totalTrialsR = hitsR+missesR+noRespsR
    
    
# trials where mouse should turn Left for reward    
    hitsL = [[] for i in range(opto_num)]  # each list is for param value (contrast, etc)  list of hits by 
    missesL = [[] for i in range(opto_num)]
    noRespsL = [[] for i in range(opto_num)]
    
    
    for i, op in enumerate(np.unique(trialOptoOnset)):
        responsesL = [trialResponse[(trialOptoOnset==op) & (trialRewardDirection==-1) & 
                                       (trialParam==val)] for val in paramVals]
        hitsL[i].append([np.sum(drs==1) for drs in responsesL])
        missesL[i].append([np.sum(drs==-1) for drs in responsesL])
        noRespsL[i].append([np.sum(drs==0) for drs in responsesL])
            
    hitsL = np.squeeze(np.array(hitsL))
    missesL = np.squeeze(np.array(missesL))
    noRespsL = np.squeeze(np.array(noRespsL))
    totalTrialsL = hitsL+missesL+noRespsL
    
        
## plotting   
 
    if plotType == 'optoByParam': 
        
        
        respAverages = []
        correctAverages = []
        
## plot as single plot with subplots 
#        fig, axs = plt.subplots(opto_num, 2, sharex='col', sharey='row', gridspec_kw={'hspace': .2, 'wspace': .5})
        
#        for i in enumerate(opto_num):
            
#        (ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = axs
        
        for i, on in enumerate(optoOnset):
            for Rnum, Lnum, Rdenom, Ldenom, title in zip([hitsR, hitsR+missesR], 
                                                         [hitsL, hitsL+missesL],
                                                         [hitsR+missesR, totalTrialsR],
                                                         [hitsL+missesL, totalTrialsL],
                                                         ['Fraction Correct Given Response', 'Response Rate']):
                
                
                
                fig, ax = plt.subplots()

                Lpoint = [num/denom if denom!=0 else 0 for (num, denom) in zip(Lnum[i],Ldenom[i])]  # workaround for division by 0 error
                ax.plot(paramVals, Lpoint , 'bo-', lw=3, alpha=.7, label='Left turning') 
                
                Rpoint = [num/denom if denom!=0 else 0 for (num, denom) in zip(Rnum[i],Rdenom[i])]
                ax.plot(paramVals, Rpoint, 'ro-', lw=3, alpha=.7, label='Right turning')

                ax.plot(paramVals, (Lnum[i]+Rnum[i])/(Ldenom[i]+Rdenom[i]), 'ko--', alpha=.5, label='Combined average') 
               
                
                if title == 'Fraction Correct Given Response':
                    correctAverages.append((Rnum[i]+Lnum[i])/(Rdenom[i]+Ldenom[i]))
                elif title == 'Response Rate':
                    respAverages.append((Rnum[i]+Lnum[i])/(Rdenom[i]+Ldenom[i]))
                
                
                showTrialN=True           
                if showTrialN==True:
                    for x,Rtrials,Ltrials in zip(paramVals,Rdenom[i], Ldenom[i]):
                        for y,n,clr in zip((1.05,1.1),[Rtrials, Ltrials],'rb'):
                            fig.text(x,y,str(n),transform=ax.transData,color=clr,fontsize=10,
                                     ha='center',va='bottom')
            
                if param == 'targetContrast':
                    pval = 'Target Contrast'
                elif param == 'targetLength':
                    pval = 'Target Length'
                elif param == 'soa':
                    pval = 'SOA'
                    
                xlab = pval
                
                formatFigure(fig, ax, xLabel=xlab, yLabel=title)
                
                value = np.round(int((on/framerate)*1000))
                
                if on >= 0:
                    fig.suptitle(('(' + mouse + ',   ' + date + ')    opto onset = ' + str(value)) + 'ms', fontsize=13)
                else:
                    fig.suptitle(('(' + mouse + ',   ' + date + ')     ' + 'No opto'), fontsize=13)
                 
                xticks = paramVals
                ax.set_xticks(xticks)
                ax.set_xticklabels(list(paramVals))

                if param=='targetLength':
                    ax.set_xlim([-5, paramVals[-1]+1])
                elif param=='targetContrast':
                    ax.set_xlim([0, 1.05])
                                    
                ax.set_ylim([0,1.05])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                plt.subplots_adjust(top=0.86, bottom=0.148, left=0.095, right=0.92, hspace=0.2, wspace=0.2)
                plt.legend(loc='best', fontsize='small', numpoints=1) 
                        

# create combined average opto plot
# need to specify if plotting response or correct 
                
#        if plotType == 'combined_opto':
      
        fig, ax = plt.subplots()
        for resp, color, al, lbl in zip(respAverages, ['k', 'b', 'b', 'b'], [1,.25,.5,1], optoOnset):
            if lbl==-1:
                label = 'No Opto'
            else:
                lbl = np.round(int((lbl/framerate)*1000))
                label = lbl.astype(str) +  ' ms onset'
                
            ax.plot(paramVals, resp, (color+'o-'),  lw=3, alpha=al, label=label)
            
        if showTrialN==True:
            for x,trials in zip(paramVals,(sum(totalTrialsR) + sum(totalTrialsL))):
                fig.text(x,1.05,str(trials),transform=ax.transData,color='k',fontsize=10,
                         ha='center',va='bottom')
    
        
        formatFigure(fig, ax, xLabel=xlab, yLabel="Response Rate")
## this yLabel may change if plotting respAverage or correctAverage

        
        xticks = paramVals
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(paramVals))

        if param=='targetLength':
            ax.set_xlim([-5, paramVals[-1]+1])
        elif param=='targetContrast':
            ax.set_xlim([0, 1.05])
        fig.suptitle('(' + mouse + ',   ' + date + ')      Combined Opto' , fontsize=13)
           
        ax.set_ylim([0,1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        plt.subplots_adjust(top=0.88, bottom=0.108, left=0.095, right=0.92, hspace=0.2, wspace=0.2)
        plt.legend(loc='best', fontsize='small', numpoints=1) 
                



## plotting the contrast levels against the opto on the x-axis -  need to specify if plotting response or correct 
        
        
        fig, ax = plt.subplots()
        
        alphaLevels = [.3,.4,.6,1]
        for resp, al, lbl in zip(np.transpose(respAverages), alphaLevels, paramVals):   #shades of gray
            
            lbl = np.round(int(lbl*100))
            label = lbl.astype(str) +  '% contrast'
            
            ax.plot(optoOnset, resp, 'ko-',  lw=3, alpha=al, label=label)
            
 # 4 nums over each line condition, in the same color as the contrast level                   
        if showTrialN==True:
            for x,trials in zip(optoOnset, ((totalTrialsR) + (totalTrialsL))):
                for y,n, al in zip((1.05,1.1,1.15,1.2),trials, alphaLevels):
                    fig.text(x,y,str(n),transform=ax.transData,color='k', alpha=al, fontsize=10,
                             ha='center',va='bottom')
                    
        formatFigure(fig, ax, xLabel='Optogenetic Onset (ms)', yLabel="Response Rate")
        ## this yLabel may change if plotting respAverage or correctAverage
        
        xlabls = [np.round(int((on/framerate)*1000)) for on in optoOnset]
        
        ax.set_xlim([-2, optoOnset[-1] +1])

        xticks = optoOnset
        ax.set_xticks(xticks)
        xlabls[0] = 'No\nopto'
        ax.set_xticklabels(xlabls)

        fig.suptitle('(' + mouse + ',   ' + date + ')      Combined Contrast' , fontsize=13)
           
        ax.set_ylim([0,1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        plt.subplots_adjust(top=0.825, bottom=0.133, left=0.1, right=0.91, hspace=0.2, wspace=0.2)
        plt.legend(loc='best', fontsize='small', numpoints=1) 
                
                