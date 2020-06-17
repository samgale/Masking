# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:26:33 2020

@author: chelsea.strawder

Plot distributions of outcome time and intiation time during training 
to monitor biases


"""

from dataAnalysis import import_data, create_df, get_dates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import numpy as np


def plot_outcomes_byside(data):
        
    df = create_df(data)
    
    
    rightTurns = df[(df['trialLength_ms']!=df['trialLength_ms'].max()) & (df['rewDir']==1) & (df['ignoreTrial']==False)]
    leftTurns = df[(df['trialLength_ms']!=df['trialLength_ms'].max()) & (df['rewDir']==-1) & (df['ignoreTrial']==False)]
    
    corrR = rightTurns['outcomeTime_ms'][rightTurns['resp']==1]
    incorrectR = rightTurns['outcomeTime_ms'][rightTurns['resp']==-1]
    
    corrL = leftTurns['outcomeTime_ms'][leftTurns['resp']==1]
    incorrectL = leftTurns['outcomeTime_ms'][leftTurns['resp']==-1]
    
    date = get_dates(df)
    mouse = df.mouse
    xMax = data['maxResponseWaitFrames'][()] * 1000/df.framerate
    
# only plotting correct     
    fig, axes = plt.subplots(2, sharex=True, sharey=True)
        
    bins = np.linspace(200, xMax, ((xMax-200)/50)+1)
    
    fig.suptitle(mouse + '   ' + date)
    axes[0].hist(corrR, weights=np.ones(len(corrR)) / len(corrR), color='r', bins=bins)
    axes[0].set(ylabel = '% Trials, Right Turning')
    axes[0].set_title('Outcome Time for correct choices by side (ms)')
       
    axes[1].hist(corrL, weights=np.ones(len(corrL)) / len(corrL), color='b', bins=bins)
    axes[1].set(ylabel='% Trials, Left Turning')
    axes[1].set(xlabel='Outcome Time (ms)')
    
    for ax in axes.flat:
        ax.set_xlim(left=200)
        
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))





    # annotate turning firection, rather than y label??
    # need to normalize by frequency?? some mice responding more to one side, makes hists taller    
    
# plotting corr and incorrect    
#    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
#    
#    fig.suptitle('Trial Outcome Time by side (ms)')
#    axes[0,0].hist(corrR, color='r')
#    axes[0,0].set_title('Right Correct')
#    
#    axes[0,1].hist(incorrectR, color='k')
#    axes[0,1].set_title('Right Incorrect')
#    
#    axes[1,0].hist(corrL, color='b')
#    axes[1,0].set_title('Left Correct')
#    
#    plt.xlabel('Outcome Time (ms)')
#    
#    axes[1,1].hist(incorrectL, color='k')
#    axes[1,1].set_title('Left Incorrect')
#     
#    for ax in axes.flat:
#        axes[0,0].set(xlabel='Outcome Time (ms)', ylabel='Number of trials')
#        ax.set_xlim(left=0)
#    
#    # Hide x labels and tick labels for top plots and y ticks for right plots.
#    for ax in axes.flat:
#        ax.label_outer()
    
