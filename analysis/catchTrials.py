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

def catch_trials(d, xlim='auto', ylim='auto', plot_ignore=False):
            
        
    df = create_df(d)
    
    monitorSize = d['monSizePix'][0] 
    normRewardDist = d['wheelRewardDistance'][()] if 'wheelRewardDistance' in d.keys() else d['normRewardDistance'][()]

    ylabel = 'Wheel Distance Turned (mm)' if 'wheelRewardDistance' in d.keys() else 'Wheel Position'
    wheelRad = d['wheelRadius'][()]
#    rewThreshold = normRewardDist if 'wheelRewardDistance' in d.keys() else normRewardDist*monitorSize
    maxResp = d['maxResponseWaitFrames'][()]
    trialRew = d['trialResponseDir'][:]
    closedLoop = d['openLoopFramesFixed'][()]
    framerate = df.framerate
    
    catchTrials = [i for i, row in df.iterrows() if row.isnull().any()]
    catchRew = [i for i in catchTrials if df.loc[i, 'trialLength_ms'] < np.max(df['trialLength_ms'])]
    
    noRew = [i for i in catchTrials if i not in catchRew]
    
    moveR = [i for i in catchTrials if trialRew[i]==1]
    moveL = [i for i in catchTrials if trialRew[i]==-1]
    ignore = [i for i in catchTrials if df.loc[i, 'ignoreTrial']==True]
    
    if xlim=='auto':
        time = np.arange(maxResp+closedLoop)/framerate
    else:
        time = np.arange(xlim[1]*framerate)/framerate
    
    
    fig, ax = plt.subplots()
    
    for i in catchTrials:
        stim = df.loc[i, 'stimStart']
        start = df.loc[i, 'trialStart']
        ind = stim - start 
        wheel = np.cumsum(df.loc[i, 'deltaWheel'][ind:]*wheelRad)
        wheel = wheel[:len(time)]
        
        if i in ignore:
           pass
        
        elif i in catchRew and i not in ignore:   # moved past reward threshold within the trial time
            ax.plot(time, wheel, c='c', alpha=.6, label="Reward Trial" if "Reward Trial"\
                    not in plt.gca().get_legend_handles_labels()[1] else '')  
#            ax.plot(wheel[])  # plotting "rewards"
            
        else:   # no response trials
            ax.plot(time, wheel, c='k', alpha=.2)
    
    ylim = ax.get_ylim()
    
    ax.vlines((closedLoop/framerate), ylim[0], ylim[1], ls='--', color='g', lw=3, label='Start Closed Loop')
    ax.vlines((maxResp + closedLoop)/framerate, ylim[0], ylim[1], ls='--', color='b', alpha=.5, lw=2, label='Max Response Wait Frame')

    if xlim=='auto':
        ax.set_xlim(0, (maxResp+closedLoop+(maxResp/2)/framerate)
    else:
        ax.set_xlim(xlim[0], xlim[1])
    
    formatFigure(fig, ax, title="Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
    
    date = get_dates(df)
    
    plt.suptitle(df.mouse + '  ' + date)
    plt.legend(loc='best', fontsize='small', numpoints=1) 
    
    ignored_counts = df['rewDir'].isnull().groupby(df['ignoreTrial']).sum()   #counting ignore trials for catch trials
    
        
    print('\n')
    print('Prob catch trial: ' + str(d['probCatch'][()]))
    print('Total catch: ' + str(len(catchTrials)))
    print('Ignored (early move): ' + str(int(ignored_counts[1])))
    print('Turn R: ' + str(len([i for i in moveR if i not in ignore])))
    print('Turn L: ' + str(len([j for j in moveL if j not in ignore])))
    print('No response: ' + str(len([k for k in noRew if k not in ignore])))
   


    if plot_ignore==True:   # plot of ignored trials 

        fig, ax = plt.subplots()
    
        for i in ignore:
            stim = df.loc[i, 'stimStart']
            start = df.loc[i, 'trialStart']
            ind = stim - start 
            wheel = np.cumsum(df.loc[i, 'deltaWheel'][ind:]*wheelRad)
            wheel = wheel[:len(time)]
            
            ax.plot(time, wheel, color='orange', alpha=.3, label='Ignored'\
                        if "Ignored" not in plt.gca().get_legend_handles_labels()[1] else '')
        
        if ylim=='auto':        
            ylim = ax.get_ylim()
        else:
            ylim=[-10,10]
    
        ax.vlines((closedLoop/framerate), ylim[0], ylim[1], ls='--', color='g', 
                  lw=3, label='Start Closed Loop')
        ax.vlines((maxResp + closedLoop)/framerate, ylim[0], ylim[1], ls='--', 
                  color='b', alpha=.5, lw=2, label='Max Response Wait Frame')
    
        if xlim=='auto':
            ax.set_xlim(0, maxResp+closedLoop/framerate)
        else:
            ax.set_xlim(xlim[0], xlim[1])
        
        formatFigure(fig, ax, title="Ignored Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
        
        plt.suptitle(df.mouse + '  ' + date)
        plt.legend(loc='best', fontsize='small', numpoints=1) 
