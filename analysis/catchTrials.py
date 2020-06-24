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
    
    catchMove = [i for i in catchTrials if df.loc[i, 'trialLength_ms'] < np.max(df['trialLength_ms'])]
    
    noRew = len(catchTrials) - len(catchMove)
    
    moveR = 0
    moveL = 0
    
    for i in catchTrials:
        if trialRew[i] == 1:
            moveR += 1
        elif trialRew[i] == -1:
            moveL += 1
    
    wheelDir = []
    
    fig, ax = plt.subplots()
    plt.vlines(closedLoop, -20, 20, ls='--', color='g', label='Start Closed Loop')
    plt.vlines(maxResp + closedLoop, -20, 20, ls='--', color='b', alpha=.5, label='Max Response Wait Frame')
    
    for i in catchTrials:
        stim = df.loc[i, 'stimStart']
        start = df.loc[i, 'trialStart']
        ol = int(df.loc[i, 'openLoopFrames'])
        ind = stim - start 
        wheel = np.cumsum(df.loc[i, 'deltaWheel'][ind:]*wheelRad)
        if i not in catchMove:
            ax.plot(wheel, c='k', alpha=.2)
            
        if i in catchMove:
            ax.plot(wheel, c='c', alpha=.6, label="Reward Trial" if "Reward Trial" not in plt.gca().get_legend_handles_labels()[1] else '')  
#            ax.plot(wheel[])  # plotting "rewards"
            direction = np.argmax(abs(wheel[ol:]) >= (abs(rewThreshold + wheel[ol]))) + ol
            wheelDir.append(wheel[direction])
    
    formatFigure(fig, ax, title="Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
    
    ax.set_xticks(np.arange(0, 160, 24))
    xlabl = [np.round(i/df.framerate, 2) for i in ax.get_xticks()]
    ax.set_xticklabels(xlabl)   
    
    date = get_dates(df)
    
    plt.suptitle(df.mouse + '  ' + date)
    plt.legend(loc='best', fontsize='small', numpoints=1) 
    
    print('\n')
    print('Prob catch trial: ' + str(d['probCatch'][()]))
    print('Total catch: ' + str(len(catchTrials)))
    print('Turn R: ' + str(moveR))
    print('Turn L: ' + str(moveL))
    print('No rew: ' + str(noRew))

