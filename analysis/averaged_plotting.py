# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:22:58 2020

@author: svc_ccg

takes behavioral data from specified mice and plots an averaged response rate
and percent correct; returns 2 plots
Declare task and mice in function call
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plotting_variable_params import plot_param
from opto_masking import plot_opto_masking
from types import SimpleNamespace    
from dataAnalysis import import_data_multi, dates_to_dict, create_df
from behaviorAnalysis import formatFigure
from plotting_opto_unilateral import plot_opto_uni
import scipy.stats


def plot_avg_beh(task=None, plot_type=None, kind=None):   # call with dates_to_dict(task==task)
    '''
    mice is a dict of mouse_ids and dates - ex: {'525458':'1022', '516686':'1015'}
    task is the task type you want to average (e.g., 'opto masking')
    if plot_type=='beh', plots combined behavioral performance for task type
    if plot_type=='rxnTime', plots average reactionTime for each condition in task type
    must specify what aspect of reaction time -- 'kind'== initiation, outcome, or turning time
    initiation = how long to initiate movement for choice
    outcome = how long to the outcome of trial
    turning time = how long between intitation and outcome (i.e. how long it took them to move the wheel)
    '''
    
    

    matplotlib.rcParams['pdf.fonttype'] = 42
    # for each mouse listed, pull file and get data
    # run task-specific plotting code but get arrays back (this reduces code and helps w indexing)
    
    if 'opto' in task:
         
        folder = 'exp_opto'
        
        if task == 'opto masking':
            param = task
            func = plot_opto_masking
            
        elif task == 'opto unilateral':
            func = plot_opto_uni
            param = None
        
        elif task=='opto timing':
            func = plot_param          
            param='opto'
            
        lbl = 'Optogenetic Light Onset Relative to Target Onset (ms)'    
        task == None
            
    else:
        
        func = plot_param
        
        if task == 'targetDuration':  # 2 plots of a single percent  #^^
            param = 'targetLength'
            folder = 'exp_duration'
            lbl = 'Target Duration (ms)'
    
        elif task == 'targetContrast':  # ^^
            param='targetContrast'
            folder = 'exp_contrast'
            lbl = 'Target Contrast'
        
        elif task == 'masking':  # ^^
            param = 'soa'
            folder = 'exp_masking'

        else:
            print('No task specified')


    mice = dates_to_dict(task=task)
    files = import_data_multi(mice, folder)  # list of the hdf5 files for each mouse 
    
    assert len(mice)==len(files), 'issue with files'
    
    if plot_type=='beh':  # creates combined performance plots
    
        percents = []   #each item in percents has mouseID, dict of param vals, then dicts of vals to plot
        
        for f in files:    # calls function appropriate to params and returns array of %s  
            percents.append(func(f, param=param, ignoreNoRespAfter=None, array_only=True)) 
            
        rn = range(len(mice))
    
    
        if task == 'opto unilateral': ## still not plotting average
            
            paramVals = list(range(len(mice)))
            y = np.array([p[5] for p in percents])
            yavg = y.mean(axis=0)
            yerr = y.std(axis=0)/(4**0.5)
            
            fig, ax = plt.subplots(3,1, figsize=(8,9), facecolor='white')
            x = np.arange(len(mice))
            
            for i, y in enumerate(yavg):
                ax[i].errorbar(x, yavg[i][0], yerr=yerr[i][0], label='Move Left', color='b', lw=2)
                ax[i].errorbar(x, yavg[i][1], yerr=yerr[i][1], label='Move Right', color='r', lw=2)
                ax[i].set_title(percents[0][2][i])   
                    
                ax[i].set_xticks(x)
                ax[i].set_xticklabels(percents[0][4], fontsize=10)
                ax[i].set_xlim([-0.5,3.5])
                ax[i].set_ylim([0,1.05])
                ax[i].set_ylabel('Fraction of trials')
                for side in ('right','top'):
                    ax[i].spines[side].set_visible(False)
                    ax[i].tick_params(direction='out',top=False,right=False) 
                ax[i].legend(fontsize='small', loc=(0.71,0.71))
                
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.5)
      
            
        else:
            paramVals = [[val for key, val in percents[i][1].items() if key==param] for i in rn]
        
        
            responseRate = [[val for key, val in percents[i][1].items() if key=='Response Rate'] for i in rn]
            fractionCorr = [[val for key, val in percents[i][1].items() if key=='Fraction Correct'] for i in rn]
            maskOnly = [[val for key, val in percents[i][1].items() if key=='maskOnly'] for i in rn]
            catchTrials = [[val for key, val in percents[i][1].items() if key=='Catch Trials'] for i in rn]
                    
            if 'opto' in task:
                responseRateNoOpto = [[val for key, val in percents[i][1].items() if key=='Response Rate No Opto'] for i in rn]
                fractionCorrNoOpto = [[val for key, val in percents[i][1].items() if key=='Fraction Correct No Opto'] for i in rn]
    
            xticks = paramVals[0][0].copy()
            xticklabels = list(xticks)
           
          
          
    # response rate plotting      
        
            if task == 'opto masking':
                colors = ['k', 'c', 'b', 'm']
                plots = percents[0][1]['Trials']
                plots[-1] = 'Catch Trials' 
                
                fig, ax = plt.subplots()
                avgs = [[] for i in range(len(plots))]
                for i, r in enumerate(responseRate):
                    for e, (y, c, p) in enumerate(zip(r[0], colors, plots)):
                #                ax.plot(paramVals[0][0], y, c=c, alpha=.4, label=p)
                        avgs[e].append(y)
                        
                for e, (c,p,noOp) in enumerate(zip(colors, plots, np.nanmean(responseRateNoOpto, axis=0)[0])):
                   ax.plot(paramVals[1][0], np.nanmean(avgs[e], axis=0), lw=3, c=c, alpha=1, label=p)
                   plt.errorbar(paramVals[1][0], 
                                 np.nanmean(avgs[e], axis=0), 
                                 yerr=scipy.stats.sem(avgs[e], axis=0,nan_policy='omit'), 
                                 color=c,
                                 alpha=1,
                                 lw=2)
                    
                   plt.plot(90, noOp, color=c, marker='o')
                   plt.errorbar(90, noOp, yerr=scipy.stats.sem(responseRateNoOpto, axis=0, 
                                                                   nan_policy='omit')[0][e], color=c, lw=2)
                
                lbl = 'Optogenetic Light Onset from Target Onset (ms)' 
                
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', numpoints=1)
                
                xticks.append(90)
                xticklabels.append('no opto')
                ax.set_xlim([10, 94])
            
            else:
            
            
                fig, ax = plt.subplots()    
                for i, (x, y) in enumerate(zip(paramVals, responseRate)):
                    print(x,y)
                    ax.plot(list(x[0]),y[0], 'k-', alpha=.4, lw=1) 
                    
        #        ax.plot(paramVals[0][0], np.nanmean(responseRate, axis=0)[0], 'k-', alpha=1, lw=3)
                plt.errorbar(paramVals[0][0], np.nanmean(responseRate, axis=0)[0], 
                             yerr=scipy.stats.sem(responseRate, axis=0, nan_policy='omit')[0], 
                             color='k', alpha=1, lw=3, label='Target Only' if task=='opto timing' else None)
                
                
                if task == 'masking':
                    ax.plot(1, np.mean(maskOnly), 'ko')
                    xticks = np.insert(xticks, 0, 1)
                    xlabels = [int(np.round((tick/120)*1000)) for tick in paramVals[0][0]]
                    xlabels[-1] =  'target\nonly'
                    xticklabels = np.insert(xlabels, 0, 'mask\nonly')
                    lbl = 'Mask Onset from Target Onset (ms)'
                    ax.set_xlim([.5,8])
                
                
                if task == 'opto timing':

                    xticklabels = [int(np.round(((tick-2)/120)*1000)) for tick in xticklabels]
                    xticklabels[-1] =  'no opto'
                    
                    for x in catchTrials:
                        ax.plot(paramVals[0][0], x[0], color='m', alpha=.3, lw=1)
                    plt.errorbar(paramVals[0][0], np.nanmean(catchTrials, axis=0)[0], 
                                 yerr=scipy.stats.sem(catchTrials, axis=0, nan_policy='omit')[0], 
                                 color='m', alpha=.6, lw=3, label='Catch Trials')
                    ax.set_xlim([3, 15])
                   
                
                
                if task=='targetContrast':
                    ax.set_xlim([.1,1.1])
                    
                    
                if task == 'targetDuration':
                    ax.set_xlim([0, 105])
                
                
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
            plt.title('Average Response Rate across mice')
            ax.set_ylim([0,1.05])
            
            formatFigure(fig, ax, xLabel=lbl, yLabel='Response Rate')  
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.92, hspace=0.2, wspace=0.2)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if len(by_label)>1:
                plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
        
            
            
    # fraction correct plotting
            if task == 'opto masking':
               
                fig, ax = plt.subplots()
                avgs = [[] for i in range(len(plots))]
                
                for i, r in enumerate(fractionCorr):
                    for e, (y, c, p) in enumerate(zip(r[0], colors, plots)):
                        avgs[e].append(y)
                    
                for e, (c,p, noOp) in enumerate(zip(colors[1:3], plots[1:3], 
                                               np.nanmean(fractionCorrNoOpto, axis=0)[0][1:3])):
                    ax.plot(paramVals[0][0], np.nanmean(avgs[e+1], axis=0), lw=3, c=c, alpha=1, label=p)
                    
                    plt.errorbar(paramVals[0][0], 
                                 np.nanmean(avgs[e+1], axis=0), 
                                 yerr=scipy.stats.sem(avgs[e+1], axis=0,nan_policy='omit'), 
                                 color=c,
                                 alpha=1,
                                 lw=2)
                    
                  #  plt.plot(90, noOp, color=c, marker='o')
                    plt.errorbar(90, noOp, yerr=scipy.stats.sem(fractionCorrNoOpto, axis=0, 
                                 nan_policy='omit')[0][e+1], color=c, lw=2)
                
                ax.set_xticks(xticks)
        
                xticklabels.append('no opto')
        
                ax.set_xlim([10, 94])  
                
                lbl = 'Optogenetic Light Onset from Target Onset (ms)' 
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', numpoints=1)
        
                ax.set_xlim([10, 94])
                ax.set_xticklabels(xticklabels)
            
            
            
            else:
         
             
                fig, ax = plt.subplots()    
                for i, (x, y) in enumerate(zip(paramVals, fractionCorr)):
            
                    ax.plot(x[0],y[0], 'k-', alpha=.4, lw=1)
                plt.errorbar(paramVals[0][0], np.nanmean(fractionCorr, axis=0)[0], 
                             yerr=scipy.stats.sem(fractionCorr, axis=0, nan_policy='omit')[0], color='k', alpha=1, lw=3)
                
                
                ax.set_xticks(paramVals[0][0])   
                    
                if task == 'masking':
                    ax.set_xticklabels(xlabels[:-1])
                    ax.set_xlim([.5, 8])
                    lbl = 'Mask Onset from Target Onset (ms)'
                
                if task == 'opto timing':
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlim([3,15])
                     
                if task == 'targetContrast':
                    ax.set_xlim([.1,1.1])
                
                if task == 'targetDuration':
                    ax.set_xlim([0, 105])
                        
            plt.title('Average Fraction Correct Given Response across mice')
            ax.set_ylim([.4,1.02])
        
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        
            formatFigure(fig, ax, xLabel=lbl, yLabel='Fraction Correct')   
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.92, hspace=0.2, wspace=0.2)
        
      
        
        
    elif plot_type=='rxnTime':  # creates plots of reaction Times
        
        dict1 = {}
        for (m, f) in zip(mice.keys(), files):
            dict1['df_{}'.format(m)] = create_df(f)
         
        if 'opto' in task:
            param='optoOnset'
        
        if task=='targetDuration':
            param='targetDuration'
        
        compiledTimes = []
        targetOnlyTimes = [[] for i in range(len(mice))]
        
        for k, df in enumerate(dict1.items()):
            
            
            paramVals = np.unique(df[1][param])
            paramVals = paramVals[:5]
            times = [[] for i in range(len(paramVals))]
            
            filtered_df = df[1][(df[1]['ignoreTrial']==False) & (df[1]['resp']==1)]
            
            for i, p in enumerate(paramVals):
                for j in range(len(filtered_df)):
                    if np.round(filtered_df.iloc[j][param], 2)==np.round(p,2):
                        times[i].append(filtered_df.iloc[j]['trialLength_ms'])
            
            xlist = filtered_df['trialLength_ms'][filtered_df['trialType']=='targetOnly']
            targetOnlyTimes[k].append(xlist.values)
            
            compiledTimes.append(times)
            
            if kind == 'initiation':
                pass
                
                
                
                ## last soa is target only

                ax.set_xticks(list(paramVals[1:]))
                xticklabels = list(np.round(list(paramVals[1:-1]),2))
                xticklabels.append('Target Only')
                ax.set_xticklabels(xticklabels)
                
            elif kind == 'outcome':
                
                paramVals = paramVals[1::]
                tarOnly = paramVals[-1] + 5
                
                
                fig, ax = plt.subplots()
                for i in compiledTimes:
                    ax.plot(paramVals, [np.mean(t) for t in i][1::], c='k', alpha=.4)
                for l, _ in enumerate(mice):
                    plt.plot(tarOnly, np.mean(targetOnlyTimes[l]), 'ko', alpha=.4)
                
                xticks = list(np.round(paramVals))
                xticks.append(tarOnly)
                ax.set_xticks(xticks)
                xticks[-1] = 'Target\n Only'
                ax.set_xticklabels(xticks)
                lbl = 'Mask Onset from Target Onset (ms)'
                formatFigure(fig, ax, xLabel=lbl, yLabel='Time (ms)')   
                
                            
                l1 = [[] for i in compiledTimes[0]]
                errlist = [[] for i in compiledTimes[0]]
                for i, _ in enumerate(compiledTimes):
                    for j, _ in enumerate(compiledTimes[0]):
                        l1[j].append(np.nanmean(compiledTimes[i][j]))
                        for x in compiledTimes[i][j]:
                            errlist[j].append(x)
                
                err = [scipy.stats.sem(errlist[i]) for i,_ in enumerate(errlist)]
                avg = [np.mean(errlist[i]) for i, _ in enumerate(errlist)]
                
                p=[]
                for i in targetOnlyTimes:
                    for j in i:
                        for x in j:
                            p.append(x)
                
                
                plt.errorbar(paramVals, avg[1::], 
                             yerr=err[1::], 
                             color='k', alpha=.6, lw=3, label='Average')
                plt.errorbar(tarOnly, np.mean(p), yerr=scipy.stats.sem(p), color='k', alpha=.6, lw=3)


            elif kind == 'turning time':
                pass
            

#corr avg
#[354.6433335369827, 344.8650784725346, 336.3205280609224, 325.6390884952466]
#


















