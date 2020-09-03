# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:20:20 2019

@author: svc_ccg
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from dataAnalysis import ignore_after, get_dates, create_df

"""
plots the choices (in order) over the length of a session

change this to create a df using dataAnalysis and the column of nogo turning?

"""

def plot_session(data, ion=True, ignoreNoRespAfter=10):
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.style.use('classic')

    
    if ion==False:
        plt.ioff()
    else:
        plt.ion()
    
    d=data

    df = create_df(d)
    framerate = df.framerate
    mouse = df.mouse
    date = get_dates(df.date)
    end = len(df)
    
    for i in range(len(df)) :
        if ~np.isfinite(df.loc[i, 'resp']):
            print(i)
            df.loc[i, 'resp']=0   # this helps with plotting cumsum
            
    df['CumPercentCorrect'] = df['resp'].cumsum()
    
    endAnalysis = ignore_after(d, ignoreNoRespAfter)
    
    
    rightCorr = df[(df['resp']==1) & (df['rewDir']==1)]
    rightMiss = df[(df['resp']==-1) & (df['rewDir']==1)]
    rightNoResp = df[(df['resp']==0) & (df['rewDir']==1)]

    
    leftCorr = df[(df['resp']==1) & (df['rewDir']==-1)]
    leftMiss = df[(df['resp']==-1) & (df['rewDir']==-1)]
    leftNoResp = df[(df['resp']==0) & (df['rewDir']==-1)]
    
    catchTrials = df[(df['trialType']=='catch') | (df['trialType']=='catchNoOpto')]

    
    fig, ax = plt.subplots(figsize=[9.75, 6.5])
    
    ax.plot(df['CumPercentCorrect'], 'k-')
    ax.plot(rightCorr['CumPercentCorrect'], 'r^', ms=10, label="right correct")
    ax.plot(leftCorr['CumPercentCorrect'], 'b^', ms=10, label="left correct")
    ax.plot(rightMiss['CumPercentCorrect'], 'rv', ms=10, label="right miss")
    ax.plot(leftMiss['CumPercentCorrect'], 'bv', ms=10, label="left miss")
    ax.plot(rightNoResp['CumPercentCorrect'], 'o', mec='r', mfc='none',  ms=10, label="right no response")
    ax.plot(leftNoResp['CumPercentCorrect'], 'o', mec='b', mfc='none', ms=10, label="left no response")
    ax.plot(catchTrials['CumPercentCorrect'], '|', color='k', ms=10)
    
    

#    for mask,i,corr in zip(df['mask'], df.index, df['CumPercentCorrect']):
#        if mask>0:
#            print(mask, i, corr)
#            plt.axvline(x=i, ymin=-100, ymax=300, c='k', ls='--', alpha=.5)
#            ax.annotate(str(mask), xy=(i,corr), xytext=(0, 20), textcoords='offset points', fontsize=8)
            
    if endAnalysis[0] != end:
        plt.vlines(endAnalysis[1], ax.get_ylim()[0], ax.get_ylim()[1], 'k', ls='--', 
                   label='End Analysis' if 'End Analysis' not in plt.gca().get_legend_handles_labels()[1] else '')
            
    plt.suptitle(mouse + date)
    plt.title('Choices over the Session')
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Time in session (min)')
    
    fig.set_facecolor('w')
    
    plt.legend(loc="best", fontsize='medium', numpoints=1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91, bottom=0.1, left=0.075, right=0.985, hspace=0.2, wspace=0.2)
    ax.margins(x=0.01, y=.01)
    labels = [str(np.round(int((ind/framerate)/60))) for ind in ax.get_xticks()]
    ax.set_xticklabels(labels)

