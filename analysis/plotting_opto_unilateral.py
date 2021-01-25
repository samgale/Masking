# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:35:50 2020

@author: svc_ccg
"""

from dataAnalysis import import_data, get_dates, ignore_after
from behaviorAnalysis import formatFigure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_opto_uni(data, param=None, ignoreNoRespAfter=0, masking=False, array_only=False):

    matplotlib.rcParams['pdf.fonttype'] = 42

    
    d = data
    end = ignore_after(d, ignoreNoRespAfter)
    if type(end) != int:
        end = end[0]
            
    mouse_id = d['subjectName'][()]

    trialType = d['trialType'][:end]
    targetContrast = d['trialTargetContrast'][:end]
    optoChan = d['trialOptoChan'][:end]
    optoOnset = d['trialOptoOnset'][:end]
    rewardDir = d['trialRewardDir'][:end]
    response = d['trialResponse'][:end]
    responseDir = d['trialResponseDir'][:end]
    
    if masking==True:
        maskOnset= d['trialMaskOnset'][:end]
        masking = maskOnset>0

    goLeft = rewardDir==-1
    goRight = rewardDir==1
    catch = np.isnan(rewardDir)
    
    noOpto = np.isnan(optoOnset)
    optoLeft = optoChan[:,0] & ~optoChan[:,1]
    optoRight = ~optoChan[:,0] & optoChan[:,1]
    optoBoth = optoChan[:,0] & optoChan[:,1]
    
   
    
# plot resps to unilateral opto by side and catch trials
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(4)
    for side,lbl,clr in zip((np.nan,-1,1),('no response','move left','move right'),'kbr'):
        n = []
        y = []
        for opto in (noOpto,optoLeft,optoRight,optoBoth):
            ind = catch & opto
            n.append(ind.sum())
            if np.isnan(side):
                y.append(np.sum(np.isnan(responseDir[ind]))/n[-1])
            else:
                y.append(np.sum(responseDir[ind]==side)/n[-1])
        ax.plot(x,y,clr,lw=2, marker='o',label=lbl)
    for tx,tn in zip(x,n):
        fig.text(tx,1.05,str(tn),color='k',transform=ax.transData,va='bottom',ha='center',fontsize=8)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('no\nopto','opto\nleft','opto\nright','opto\nboth'))
    ax.set_xlim([-0.5,3.5])
    ax.set_ylim([0,1.05])
    ax.set_ylabel('Fraction of catch trials')
    ax.legend(fontsize='small', loc='best')
    plt.tight_layout()
    plt.subplots_adjust(top=.920)
    fig.text(0.525,0.99,'Catch trial movements',va='top',ha='center')
    formatFigure(fig, ax) 
   
    
    
    fig = plt.figure(figsize=(8,9))
    gs = matplotlib.gridspec.GridSpec(8,1, hspace=.7)
    x = np.arange(4)
    
    returnArray = [[],[],[]]

    for j,contrast in enumerate([c for c in np.unique(targetContrast) if c>0]):
        for i,(trials,trialLabel) in enumerate(zip((goLeft,goRight,catch),('Right Stimulus','Left Stimulus','No Stimulus'))):
            if i<2 or j==0:
                ax = fig.add_subplot(gs[i*3:i*3+2,j])
                for resp,respLabel,clr,ty in zip((-1,1),('move left','move right'),'br',(1.05,1.1)):
                    n = []
                    y = []
                    for opto in (noOpto,optoLeft,optoRight,optoBoth):
                        ind = trials & opto
                        if trialLabel != 'No Stimulus':
                            ind = trials & opto & (targetContrast==contrast)
                        n.append(ind.sum())
                        y.append(np.sum(responseDir[ind]==resp)/n[-1])
                        
                    returnArray[i].append(y)

                    ax.plot(x,y,clr,marker='o', lw=2, label=respLabel)
                for tx,tn in zip(x,n):
                    fig.text(tx,ty,str(tn),color='k',transform=ax.transData,va='bottom',ha='center',fontsize=8)
                title = trialLabel if trialLabel=='No Stimulus' else trialLabel+', Contrast '+str(contrast)
                fig.text(1.5,1.25,title,transform=ax.transData,va='bottom',ha='center',fontsize=12)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(x)
                xticklabels = ('no\nopto','opto\nleft','opto\nright','opto\nboth')# if i==2 else []
                ax.set_xticklabels(xticklabels, fontsize=10)
                ax.set_xlim([-0.5,3.5])
                ax.set_ylim([0,1.05])
                if j==0:
                    ax.set_ylabel('Fraction of trials')
                if i==0 and j==0:
                    ax.legend(fontsize='small', loc=(0.71,0.71))
#    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.3)
    formatFigure(fig, ax)              
                    
    if array_only==True:
        return(mouse_id, 
               ['each list is stim L/R?none, and each list inside is move L/R'],
               ['Right Stimulus, 40% contrast', 'Left Stimulus, 40% contrast', 'No Stim'],
               ['Move Left', 'Move Right'],
               ['No Opto', 'Opto Left', 'Opto Right', 'Opto Both'],
               returnArray)
    
    
    
