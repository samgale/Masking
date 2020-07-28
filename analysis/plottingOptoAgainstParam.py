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
    optoOnset = d['optoOnset'][:]   # original opto onsets 
    trialOptoOnset = d['trialOptoOnset'][:end]
    
    for i, trial in enumerate(trialOptoOnset):  # replace nans (no opto) with -1
        if ~np.isfinite(trial):
             noOpto = optoOnset[-1] + np.median(np.round(np.diff(optoOnset)))
             trialOptoOnset[i] = noOpto

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
    optoOn = np.unique(trialOptoOnset)   # opto onsets with no opto encoded as max(opto) + avg opto diff
    if param == 'targetLength' or param == 'targetContrast':
        paramVals = paramVals[paramVals>0]
        
    param_num = len(np.unique(paramVals))



    # each list is for an opto onset, and the values within are by paramValue  
    #  ex:  list1 = opto Onset 0, within are trials for each contrast level [0, .2, .4, 1]    
    hitsR = [[] for i in range(opto_num)]
    missesR = [[] for i in range(opto_num)]
    noRespsR = [[] for i in range(opto_num)]
     
    for i, op in enumerate(optoOn):
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
    
    
    for i, op in enumerate(optoOn):
        responsesL = [trialResponse[(trialOptoOnset==op) & (trialRewardDirection==-1) & 
                                       (trialParam==val)] for val in paramVals]
        hitsL[i].append([np.sum(drs==1) for drs in responsesL])
        missesL[i].append([np.sum(drs==-1) for drs in responsesL])
        noRespsL[i].append([np.sum(drs==0) for drs in responsesL])
            
    hitsL = np.squeeze(np.array(hitsL))
    missesL = np.squeeze(np.array(missesL))
    noRespsL = np.squeeze(np.array(noRespsL))
    totalTrialsL = hitsL+missesL+noRespsL
    
    catch = [[[] for i in range(opto_num)]]*2
    for trial, turn, op in zip(trialParam, d['trialResponseDir'][:], trialOptoOnset):
        if trial==0:
            pass
            
            
    
    
        
## plotting   
 

        
        
   
## plot as single plot with subplots 
#        fig, axs = plt.subplots(opto_num, 2, sharex='col', sharey='row', gridspec_kw={'hspace': .2, 'wspace': .5})
    
#        for i in enumerate(opto_num):
        
#        (ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = axs
    respAverages = []
    correctAverages = []
    
    
    for i, on in enumerate(optoOn):
        for Rnum, Lnum, Rdenom, Ldenom, title in zip([hitsR, hitsR+missesR], 
                                                     [hitsL, hitsL+missesL],
                                                     [hitsR+missesR, totalTrialsR],
                                                     [hitsL+missesL, totalTrialsL],
                                                     ['Fraction Correct Given Response', 'Response Rate']):
            
            
            
            fig, ax = plt.subplots()

            Lpoint = [num/denom if denom!=0 else 0 for (num, denom) in zip(Lnum[i],Ldenom[i])]  
                # workaround for division by 0 error
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
            
            value = np.round(int((on/framerate)*1000))   # onset expressed in ms
            
            if on == noOpto:
                fig.suptitle(('(' + mouse + ',   ' + date + ')     ' + 'No opto'), fontsize=13)
            else:
                fig.suptitle(('(' + mouse + ',   ' + date + ')    opto onset = ' + str(value)) + 'ms', fontsize=13)
             
            xticks = paramVals
            ax.set_xticks(xticks)
            ax.set_xticklabels(list(paramVals))

            if param=='targetLength':
                ax.set_xlim([-5, paramVals[-1]+1])
            elif param=='targetContrast':
                ax.set_xlim([.2, 1.05])  #############################FIX 
                                
            ax.set_ylim([0,1.05])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.subplots_adjust(top=0.86, bottom=0.123, left=0.1, right=0.92, hspace=0.2, wspace=0.2)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
                    

# create combined average opto plot
# need to specify if plotting response or correct 
                
#        if plotType == 'combined_opto':
      
    for avgs, yLbl in zip([respAverages, correctAverages], ['Response Rate', 'Fraction Correct']):
    
        fig, ax = plt.subplots()
        colors = []   
        for i in range(opto_num-1):
            colors.append('b')
        colors.append('k')  # no opto condition
        alphas = np.linspace(.2,1, opto_num)   ### need to find a different color system
        
        for resp, color, al, lbl in zip(avgs, colors, alphas, optoOn):
            if lbl==noOpto:
                label = 'No Opto'
            else:
                lbl = np.round(int((lbl/framerate)*1000))
                label = lbl.astype(str) +  ' ms onset'
                
            ax.plot(paramVals, resp, (color+'o-'),  lw=3, alpha=al, label=label)
        
        if showTrialN==True:
            text_spacing = [1.05]
            for _ in range(opto_num-1):
                text_spacing.append(np.round(text_spacing[-1]+.05,3))
                
            for x,trials in zip(paramVals,(np.transpose(totalTrialsR) + np.transpose(totalTrialsL))):
                for y, n, al, clr in zip(text_spacing, trials, alphas, colors):
                    fig.text(x, y, str(n), transform= ax.transData,  color=clr, alpha=al, fontsize=10, 
                             ha='center',va='bottom')
        
        
        formatFigure(fig, ax, xLabel=xlab, yLabel=yLbl)
        
        xticks = paramVals
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(paramVals))

        if param=='targetLength':
            ax.set_xlim([-5, paramVals[-1]+1])
        elif param=='targetContrast':
            ax.set_xlim([paramVals[0]-.2, 1.05]) #############################FIX 
            
        fig.suptitle('(' + mouse + ',   ' + date + ')      Combined Opto Onset vs Contrast' , fontsize=13)
           
        ax.set_ylim([0,1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        plt.subplots_adjust(top=(.915 - (.025*opto_num)), bottom=0.108, left=0.1, right=0.945, hspace=0.2, wspace=0.2)
        plt.legend(loc='best', fontsize='small', numpoints=1) 
                


    
## plotting the contrast levels against the opto on the x-axis -  need to specify if plotting response or correct 
## shades of black          
            
        fig, ax = plt.subplots()
        
        alphaLevels = np.linspace(.2,1, param_num)
        
        optoList = list(map(int, optoOn))
       
        
        for resp, al, lbl in zip(np.transpose(avgs), alphaLevels, paramVals):   #shades of gray
            
            lbl = np.round(int(lbl*100))
            label = lbl.astype(str) +  '% contrast'
            
            ax.plot(optoList, resp, 'ko-',  lw=3, alpha=al, label=label)

            showTrialN=True
            if showTrialN==True:
                text_spacing = [1.05]
                for _ in range(param_num-1):
                    text_spacing.append(np.round(text_spacing[-1]+.05,3))
                
                for x,trials in zip(optoList, ((totalTrialsR + totalTrialsL))):
                    for y, n, al in zip(text_spacing, trials, alphaLevels):
                        fig.text(x,y,str(n),transform=ax.transData,color='k', alpha=al, fontsize=10,
                                 ha='center',va='bottom')
                        
            formatFigure(fig, ax, xLabel='Optogenetic Onset (ms)', yLabel=yLbl)
            
            xlabls = [np.round(int((on/framerate)*1000)) for on in optoList]
            
            ax.set_xlim([optoList[0]-1, optoList[-1] + 1])  
    
            xticks = optoList
            ax.set_xticks(xticks)
            xlabls[-1] = 'No\nopto'
            ax.set_xticklabels(xlabls)
    
            fig.suptitle('(' + mouse + ',   ' + date + ')      Combined Contrast vs Opto onset' , fontsize=13)
               
            ax.set_ylim([0,1.05])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.subplots_adjust(top=0.815, bottom=0.133, left=0.1, right=0.945, hspace=0.2, wspace=0.2)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
                    
                        