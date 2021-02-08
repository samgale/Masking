# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:35:50 2020

@author: svc_ccg
"""

from dataAnalysis import import_data, get_dates, ignore_after
import ignoreTrials
from behaviorAnalysis import formatFigure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def plot_opto_uni(data, param=None, ignoreNoRespAfter=None, masking=False, array_only=False):

#f = fileIO.getFile('',fileType='*.hdf5')
#d = h5py.File(f,'r')   
# 
#ignoreNoRespAfter = None
    d = data

    end = ignore_after(d, ignoreNoRespAfter)
    if type(end) != int:
        end = end[0]
        
    earlyMove = ignoreTrials.ignore_trials(d)
    
    trials = [i for i in range(end) if i not in earlyMove]
            
    mouse_id = d['subjectName'][()]
    
    trialType = d['trialType'][trials]
    targetContrast = d['trialTargetContrast'][trials]
    optoChan = d['trialOptoChan'][trials]
    optoOnset = d['trialOptoOnset'][trials]
    rewardDir = d['trialRewardDir'][trials]
    response = d['trialResponse'][trials]
    responseDir = d['trialResponseDir'][trials]
    
    if masking:
        maskOnset= d['trialMaskOnset'][trials]
    
    goLeft = rewardDir==-1
    goRight = rewardDir==1
    catch = np.isnan(rewardDir)
    
    noOpto = np.isnan(optoOnset)
    optoLeft = optoChan[:,0] & ~optoChan[:,1]
    optoRight = ~optoChan[:,0] & optoChan[:,1]
    optoBoth = optoChan[:,0] & optoChan[:,1]
    
    
    # new
    if masking==True:
        fig = plt.figure(figsize=(10,10),facecolor='w')
        gs = matplotlib.gridspec.GridSpec(4,4,hspace=.7)
        x = np.arange(4)
        yticks = [0,0.5,1]
        for i,trialLbl in enumerate(('catch','targetOnly','mask','maskOnly')):
            if trialLbl in ('catch','maskOnly'):
                s = (catch,)
                slbl = ('no stim',) if trialLbl=='catch' else ('mask only',)
                h = 1
            else:
                s,slbl = ((goRight,goLeft),('target left','target right'))
                h = 0
            for j,(side,sideLbl) in enumerate(zip(s,slbl)):
                ax = fig.add_subplot(gs[i,j*2+h:j*2+h+2])
                for resp,respLbl,clr in zip((-1,1),('move left','move right'),'br'):
                    n = []
                    y = []
                    for opto in (noOpto,optoLeft,optoRight,optoBoth):
                        ind = np.in1d(trialType,(trialLbl,trialLbl+'Opto')) & side & opto
                        if any(ind):
                            n.append(ind.sum())
                            y.append(np.sum(responseDir[ind]==resp)/n[-1])
                    ls,lbl = ('--','incorrect') if (sideLbl=='target left' and resp<0) or (sideLbl=='target right' and resp>0) else ('-','correct')
                    ax.plot(x,y,clr,marker='o',linestyle=ls,label=respLbl)
                for tx,tn in zip(x,n):
                    fig.text(tx,1.05,str(tn),color='0.5',transform=ax.transData,va='bottom',ha='center',fontsize=8)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(x)
                xticklabels = ('no\nopto','opto\nleft','opto\nright','opto\nboth') if i==3 else []
                ax.set_xticklabels(xticklabels)
                ax.set_yticks([0,0.5,1])
                yticklabels = yticks if j==0 else []
                ax.set_yticklabels(yticklabels)
                ax.set_xlim([-0.5,3.5])
                ax.set_ylim([0,1.05])
                if i==0:
                    ax.set_ylabel('Fraction of trials')
                    ax.legend(fontsize=10,loc='upper right')
                lbl = sideLbl+' + mask' if trialLbl=='mask' else sideLbl
                fig.text(1.5,1.25,lbl,transform=ax.transData,va='bottom',ha='center')
    
                
    
    else:
    
    
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
        
    
    
