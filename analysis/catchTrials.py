# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:24:30 2020

@author: chelsea.strawder

Process catch trials from df 
"""

from dataAnalysis import create_df, get_dates
from behaviorAnalysis import formatFigure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype']=42

def catch_trials(d):
    
        # filter out catch trials
        # find trials where they turned past the reward threshold; which direction
        # count the trials turned R, L, or not past thresh
        # plot catch trial wheel trace 
        # To the right of the catch trial wheel plot, print the fraction of trials 
        # that were left, right, or no move past threshold.
        # and add quiescent violations  ****
        
        
    df = create_df(d)
    
    monitorSize = d['monSizePix'][0] 
    normRewardDist = d['wheelRewardDistance'][()] if 'wheelRewardDistance' in d.keys() else d['normRewardDistance'][()]
    #maxQuiescentMove = d['maxQuiescentNormMoveDist'][()]
   # sigMove = maxQuiescentMove * monitorSize
    ylabel = 'Wheel Distance Turned (mm)' if 'wheelRewardDistance' in d.keys() else 'Wheel Position'
    wheelRad = d['wheelRadius'][()]
    rewThreshold = normRewardDist if 'wheelRewardDistance' in d.keys() else normRewardDist*monitorSize
    maxResp = d['maxResponseWaitFrames'][()]
    trialRew = d['trialResponseDir'][:]
    closedLoop = d['openLoopFramesFixed'][()]
    
    catchTrials = [i for i, row in df.iterrows() if row.isnull().any()]
    catchRew = [i for i in catchTrials if df.loc[i, 'trialLength_ms'] < np.max(df['trialLength_ms'])]
    
    noRew = len(catchTrials) - len(catchRew)
    
    moveR = [i for i in catchTrials if trialRew[i]==1]
    moveL = [i for i in catchTrials if trialRew[i]==-1]
    ignore = [i for i in catchTrials if df.loc[i, 'ignoreTrial']==True]
        
    fig, ax = plt.subplots()
    plt.vlines(closedLoop, -20, 20, ls='--', color='g', lw=3, label='Start Closed Loop')
    plt.vlines(maxResp + closedLoop, -20, 20, ls='--', color='b', alpha=.5, lw=2, label='Max Response Wait Frame')

    
    for i in catchTrials:
        stim = df.loc[i, 'stimStart']
        start = df.loc[i, 'trialStart']
        ind = stim - start 
        wheel = np.cumsum(df.loc[i, 'deltaWheel'][ind:]*wheelRad)
        
        if i in ignore:
           pass
#            ax.plot(wheel, color='orange', alpha=.3, label='Ignored'\
#                    if "Ignored" not in plt.gca().get_legend_handles_labels()[1] else '')
           # if df.loc[i, ']
        
        elif i in catchRew and i not in ignore:
            ax.plot(wheel, c='c', alpha=.6, label="Reward Trial" if "Reward Trial"\
                    not in plt.gca().get_legend_handles_labels()[1] else '')  
#            ax.plot(wheel[])  # plotting "rewards"
            
        else:   # no reward and not ignore
            ax.plot(wheel, c='k', alpha=.2)
    
    formatFigure(fig, ax, title="Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
    
    xlabl = [np.round(i/df.framerate, 1) for i in ax.get_xticks()]
    ax.set_xticklabels(xlabl)   
    
    date = get_dates(df)
    
    plt.suptitle(df.mouse + '  ' + date)
    plt.legend(loc='best', fontsize='small', numpoints=1) 
    
    ignored_counts = df['rewDir'].isnull().groupby(df['ignoreTrial']).sum()   #counting ignore trials for catch trials
    
    #### applt similar counut function from percentCorrect to get L, R
    i
    
    
    print('\n')
    print('Prob catch trial: ' + str(d['probCatch'][()]))
    print('Total catch: ' + str(len(catchTrials)))
    print('Ignored (early move): ' + str(int(ignored_counts[1])))
    print('Turn R: ' + str(len([i for i in moveR if i not in ignore])))
    print('Turn L: ' + str(len([j for j in moveL if j not in ignore])))
    print('No response: ' + str(noRew))
   

