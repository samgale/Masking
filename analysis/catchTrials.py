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

def catch_trials(d, xlim='auto', ylim='auto', plot_ignore=False, arrayOnly=False, 
                 ion=True, ignoreNoRespAfter=None):
            
### figure out how to add ignoreNoRespAfter to work with this code
        
    df = create_df(d)
    framerate = df.framerate
    date = get_dates(df)
    mouse = df.mouse
    
    df = df[:-1]  # remove final trial
    monitorSize = d['monSizePix'][0] 
    normRewardDist = d['wheelRewardDistance'][()] if 'wheelRewardDistance' in d.keys() else d['normRewardDistance'][()]

    ylabel = 'Wheel Distance Turned (mm)' if 'wheelRewardDistance' in d.keys() else 'Wheel Position'
    wheelRad = d['wheelRadius'][()]
#    rewThreshold = normRewardDist if 'wheelRewardDistance' in d.keys() else normRewardDist*monitorSize
    maxResp = d['maxResponseWaitFrames'][()]
    trialResp = d['trialResponseDir'][:]
    closedLoop = d['openLoopFramesFixed'][()]
   
    
    catchTrials = df[df['trialType']=='catch'] if 'trialType' in d.keys() else df[df['rewDir'].isnull()==True]
    catch = list(catchTrials.index)
    catchMove = [i for i in catch if catchTrials.loc[i, 'trialLength_ms'] < np.max(catchTrials['trialLength_ms'])]
    
    noRew = [i for i in catch if i not in catchMove]
    
    moveR = [i for i in catch if trialResp[i]==1]
    moveL = [i for i in catch if trialResp[i]==-1]
    ignore = [i for i in catch if df.loc[i, 'ignoreTrial']==True]
    
    if xlim=='auto':
        time = np.arange((maxResp+(closedLoop*2)))/framerate
    else:
        time = np.arange(xlim[1]*framerate)/framerate
    
    ignored_counts = df['rewDir'].isnull().groupby(df['ignoreTrial']).sum()  #counting ignore trials for catch trials
    print_ignore = ignored_counts[1] if ignored_counts[0]==True else 0  

    array = ['Prob catch trial: ' + str(d['probCatch'][()]),
             ' ',
             'Total catch: ' + str(len(catchTrials)),
             'Ignored (early move): ' + str(int(print_ignore)),
             'Turn R: ' + str(len([i for i in moveR if i not in ignore])),
             'Turn L: ' + str(len([j for j in moveL if j not in ignore])),
             'No response: ' + str(len([k for k in noRew if k not in ignore]))] 
    
        
    if arrayOnly==True:
        return array
        
    else:   
        print('\n')
        for count in array:
            print(count)
            
        plt.ion()
        if ion==False:
            plt.ioff()
        
        fig, ax = plt.subplots()
        
        for i in catch:
            stim = catchTrials.loc[i, 'stimStart']
            start = catchTrials.loc[i, 'trialStart']
            ind = stim - start 
            wheel = np.cumsum(catchTrials.loc[i, 'deltaWheel'][ind:]*wheelRad)
            wheel = wheel[:len(time)]
            
            if i in ignore:
               pass
            
            elif i in catchMove and i not in ignore:   # moved past reward threshold within the trial time
                ax.plot(time, wheel, c='c', alpha=.6, 
                        label="Reward Trial" if "Reward Trial" not in plt.gca().get_legend_handles_labels()[1] else '')  
                
            else:   # no response trials
                ax.plot(time, wheel, c='k', alpha=.2)
        
        
        ylim = ax.get_ylim() if ylim=='auto' else ylim
        
        
        ax.vlines((closedLoop/framerate), ylim[0], ylim[1], ls='--', color='g', lw=3, label='Start Closed Loop')
        ax.vlines((maxResp + closedLoop)/framerate, ylim[0], ylim[1], ls='--', color='b', alpha=.5, lw=2, label='Max Response Wait Frame')
    
        if xlim=='auto':
            ax.set_xlim(0, ((maxResp+(closedLoop*2))/framerate))
        else:
            ax.set_xlim(xlim[0], xlim[1])
        
        formatFigure(fig, ax, title="Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
        
       
        
        plt.suptitle(mouse + '  ' + date)
        plt.legend(loc='best', fontsize='small', numpoints=1) 
        plt.tight_layout()
        plt.subplots_adjust(top=.9)
    
        
        if plot_ignore==True:   # plot of ignored trials 
    
            fig, ax = plt.subplots()
        
            for i in ignore:
                stim = df.loc[i, 'stimStart']
                start = df.loc[i, 'trialStart']
                ind = stim - start 
                wheel = np.cumsum(df.loc[i, 'deltaWheel'][ind:]*wheelRad)
                wheel = wheel[:len(time)]
                
                ax.plot(time, wheel, color='orange', alpha=.5, label='Ignored'\
                            if "Ignored" not in plt.gca().get_legend_handles_labels()[1] else '')
            
            if ylim=='auto':        
                ylim = ax.get_ylim()
            else:
                ylim=ylim
        
            ax.vlines((closedLoop/framerate), ylim[0], ylim[1], ls='--', color='g', 
                      lw=3, label='Start Closed Loop')
            ax.vlines((maxResp + closedLoop)/framerate, ylim[0], ylim[1], ls='--', 
                      color='b', alpha=.5, lw=2, label='Max Response')
        
            if xlim=='auto':
                ax.set_xlim(0, ((maxResp+(closedLoop*2))/framerate))
            else:
                ax.set_xlim(xlim[0], xlim[1])
            
            formatFigure(fig, ax, title="Ignored Catch Trial Wheel Traces", xLabel="Trial Length (s)", yLabel=ylabel) 
            
            plt.suptitle(df.mouse + '  ' + date)
            plt.legend(loc='best', fontsize='small', numpoints=1) 
            plt.tight_layout()
            plt.subplots_adjust(top=.9)
            
            
