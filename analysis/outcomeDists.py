# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:26:33 2020

@author: chelsea.strawder

Plot distributions of outcome time and intiation time during training 
to monitor biases


"""

from dataAnalysis import import_data, create_df
import matplotlib.pyplot as plt



def plot_outcomes_byside(data):
    

    df = create_df(data)
    
    
    rightTurns = df[(df['trialLength_ms']!=df['trialLength_ms'].max()) & (df['rewDir']==1)]
    leftTurns = df[(df['trialLength_ms']!=df['trialLength_ms'].max()) & (df['rewDir']==-1)]
    
    corrR = rightTurns['outcomeTime_ms'][rightTurns['resp']==1]
    incorrectR = rightTurns['outcomeTime_ms'][rightTurns['resp']==-1]
    
    corrL = leftTurns['outcomeTime_ms'][leftTurns['resp']==1]
    incorrectL = leftTurns['outcomeTime_ms'][leftTurns['resp']==-1]
    
    
    
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
    
    fig.suptitle('Trial Outcome Time by side (ms)')
    axes[0,0].hist(corrR, color='r')
    axes[0,0].set_title('Right Correct')
    
    axes[0,1].hist(incorrectR, color='k')
    axes[0,1].set_title('Right Incorrect')
    
    axes[1,0].hist(corrL, color='b')
    axes[1,0].set_title('Left Correct')
    
    axes[1,1].hist(incorrectL, color='k')
    axes[1,1].set_title('Left Incorrect')
    
    
    for ax in axes.flat:
        ax.set(xlabel='Outcome Time (ms)', ylabel='Number of trials')
        ax.set_xlim(left=0)
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.label_outer()
    
    
    
