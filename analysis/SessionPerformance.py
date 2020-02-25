# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:20:20 2019

@author: svc_ccg
"""

import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
from nogoData import nogo_turn

"""
plots the choices (in order) over the length of a session

"""

matplotlib.rcParams['pdf.fonttype'] = 42

def plot_session(data):
    
    d=data
    trialResponse = d['trialResponse'][()]
    trialResponseFrame = d['trialResponseFrame'][:len(trialResponse)]
    trialTargetFrames= d['trialTargetFrames'][:len(trialResponse)]   # to identify nogos 
    trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]
    maskOnset = d['trialMaskOnset'][:len(trialResponse)]
    trialMaskContrast = d['trialMaskContrast'][:len(trialResponse)]
    
    for i, trial in enumerate(trialTargetFrames):
        if trial==0:
            trialRewardDirection[i] = 0 
    
    data = zip(trialRewardDirection, trialResponse, maskOnset, trialTargetFrames, trialMaskContrast)
    
    df = pd.DataFrame(data, index=trialResponseFrame, columns=['rewardDir', 'trialResp', 'mask', 'target', 'maskCon'])
    df['CumPercentCorrect'] = df['trialResp'].cumsum()
    index = df.index
    values = df.values
    
    #function? 
    rightCorr = df[(df['trialResp']==1) & (df['rewardDir']==1)]
    leftCorr = df[(df['trialResp']==1) & (df['rewardDir']==-1)]
    nogoCorr = df[(df['trialResp']==1) & (df['rewardDir']==0)]
    
    rightMiss = df[(df['trialResp']==-1) & (df['rewardDir']==1)]
    leftMiss = df[(df['trialResp']==-1) & (df['rewardDir']==-1)]
    nogoMiss = df[(df['trialResp']==-1) & (df['rewardDir']==0)]
    
    rightNoResp = df[(df['trialResp']==0) & (df['rewardDir']==1)]
    leftNoResp = df[(df['trialResp']==0) & (df['rewardDir']==-1)]
    
    
    fig, ax = plt.subplots()
    ax.plot(df['CumPercentCorrect'], 'k-')
    ax.plot(rightCorr['CumPercentCorrect'], 'r^', ms=10)
    ax.plot(leftCorr['CumPercentCorrect'], 'b^', ms=10)
    ax.plot(rightMiss['CumPercentCorrect'], 'rv', ms=10)
    ax.plot(leftMiss['CumPercentCorrect'], 'bv', ms=10)
    ax.plot(rightNoResp['CumPercentCorrect'], 'o', mec='r', mfc='none',  ms=10)
    ax.plot(leftNoResp['CumPercentCorrect'], 'o', mec='b', mfc='none', ms=10)
    
    
    if 0 in trialTargetFrames:
    
        no_gos = nogo_turn(d, ignoreRepeats=False, returnArray=True)[0] # False bc we want to see all the trials in order 
        
        nogoMiss = pd.DataFrame(nogoMiss['CumPercentCorrect'])
        ax.plot(nogoCorr['CumPercentCorrect'], 'g^', ms=10)
        
        # set marker face fill style to reflect direction turned 
        for nogo, x, direction in zip(no_gos, nogoMiss.index, nogoMiss['CumPercentCorrect']):
            if nogo > 0:
                plt.plot(x, direction, 'gv', ms=10, markerfacecoloralt='red', fillstyle='left')  
            elif nogo < 0:
                plt.plot(x, direction, 'gv', ms=10, markerfacecoloralt='c', fillstyle='left')
    
    for mask,i,corr in zip(df['mask'], df.index, df['CumPercentCorrect']):
        if mask>0:
            print(mask, i, corr)
            plt.axvline(x=i, ymin=-100, ymax=300, c='k', ls='--', alpha=.5)
            ax.annotate(str(mask), xy=(i,corr), xytext=(0, 20), textcoords='offset points', fontsize=8)
            
        
    plt.title(str(d).split('_')[-3:-1])
    plt.show()


