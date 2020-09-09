# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:20:20 2019

@author: svc_ccg
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from dataAnalysis import ignore_after, get_dates, create_df, import_data

"""
plots the choices (in order) over the length of a session


"""

def plot_session(data, ion=True, ignoreNoRespAfter = None):
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.style.use('classic')


    if ion==False:
        plt.ioff()
    else:
        plt.ion()
  
    d = import_data()
    
    
    df = create_df(d)
    framerate = df.framerate
    mouse = df.mouse
    date = get_dates(df.date)
    end = len(df)
    
    for i in range(len(df)) :
        if ~np.isfinite(df.loc[i, 'resp']):
            df.loc[i, 'resp']=0   # this helps with plotting cumsum
            
    df['CumPercentCorrect'] = df['resp'].cumsum()
    
    endAnalysis = ignore_after(d, ignoreNoRespAfter)
    
    df.index = df['respFrame']
    
    rightTotal = df[df['rewDir']==1]
    rightCorr = rightTotal[rightTotal['resp']==1]
    rightMiss =  rightTotal[rightTotal['resp']==-1]
    rightNoResp =  rightTotal[rightTotal['resp']==0]
    
    leftTotal = df[df['rewDir']==-1]
    leftCorr = leftTotal[leftTotal['resp']==1]
    leftMiss = leftTotal[leftTotal['resp']==-1]
    leftNoResp = leftTotal[leftTotal['resp']==0]
    
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
            
    plt.suptitle(mouse + '    ' + date)
    plt.title('Choices over the Session')
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Time in session (min)')
    
    fig.set_facecolor('w')
    
    plt.legend(loc="best", fontsize='medium', numpoints=1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91, bottom=0.09, left=0.075, right=0.985, hspace=0.2, wspace=0.2)
    ax.margins(x=0.01, y=.01)
    
    labels = [str(np.round(int((ind/framerate)/60))) for ind in ax.get_xticks()]
    ax.set_xticklabels(labels)
    
 
    
#def session_stats(d, returnAs='str_array'):    #returnAs 'str_array', 'num_array', or 'print'
#
#    print(str(d) + '\n')
#    
##    df = create_df(d)
##
##    framerate = df.framerate
##    
##    if ignoreNoRespAfter is not None:
##        end = ignore_after(d, ignoreNoRespAfter)[0]
##        df = df[:end]
##    
##    sessionDuration = d['trialResponseFrame'][-1]   # in frames
##    
##    
##    catchTrials = df['catchTrial'].value_counts()[1] if d['probCatch'][()]>0 else 0
##    notCatch = df['catchTrial'].value_counts()[0]
##    
##    assert (len(df) - catchTrials) == (notCatch), "catch trial issue"
##        
##    totalTrials = notCatch
##    
##    rightTotal = df[(df['rewDir']==1) & (df['repeat']==False) & (df['catchTrial']==False)]  #ignored included
##    right = rightTotal[rightTotal['ignoreTrial']==False]   #ignore removed
##    
##    rightCorrect = len(right[right['resp']==1])
##    rightIncorrect = len(right[right['resp']==-1])
##    rightNoResp = len(right[right['resp']==0])
##    
##    
##    leftTotal = df[(df['rewDir']==-1) & (df['repeat']==False) & (df['catchTrial']==False)]  #ignored included
##    left = leftTotal[leftTotal['ignoreTrial']==False]
##    
##    leftCorrect = len(left[left['resp']==1])
##    leftIncorrect = len(left[left['resp']==-1])
##    leftNoResp = len(left[left['resp']==0])
##    
#    
#    rightTotal = len(rightTotal)
#    
#    
#    
#    ignored = df.groupby(['rewDir'])['ignoreTrial'].sum(sorted=True) # num of ignore trials by side and total
#    
#    repeats = df[(df['repeat']==True) & (df['catchTrial']==False) & (df['ignoreTrial']==False)]
#    
#    usableTrialTotal = (len(left) + len(right))   # no ignores, repeats, or catch
#    respTotal = usableTrialTotal - (rightNoResp + leftNoResp)
#    totalCorrect = (rightCorrect + leftCorrect)
#    
#    respTime = d['maxResponseWaitFrames'][()]/framerate 
#    
# 
#    array = ['Wheel Reward Dist: ' + str(d['wheelRewardDistance'][()]),
#             'Norm reward: ' + str(d['normRewardDistance'][()]),
#             'Response Window: ' + str(respTime) + ' sec',
#             'Prob go right: ' + str(d['probGoRight'][()]),
#             'Session duration (mins): ' + str(np.round(sessionDuration/framerate/60, 2)),
#             ' ',
#             'Repeats: ' + str(len(repeats)) + '/' + str(totalTrials),
#             'Total Ignore: ' + str(ignored[1]+ignored[-1]),
#             'Performance trials: ' + str(usableTrialTotal),
#             ' ',
#             'Right Ignore: ' + str(ignored[1]),
#             'Right trials: ' + str(len(right)),
#             'R % Correct: ' + str(np.round(rightCorrect/len(right), 2)),
#             'R % Incorrect: ' + str(np.round(rightIncorrect/len(right),2)),
#             'R % No Resp: ' + str(np.round(rightNoResp/len(right),2)),
#             ' ',
#             'Left Ignore: ' + str(ignored[-1]),
#             'Left trials: ' + str(len(left)),
#             'L % Correct: ' + str(np.round(leftCorrect/len(left), 2)),
#             'L % Incorrect: ' + str(np.round(leftIncorrect/len(left),2)),
#             'L % No Resp: ' + str(np.round(leftNoResp/len(left),2)),
#             ' ',
#             'Total Correct, given Response: ' + str(np.round((leftCorrect+rightCorrect)/respTotal,2)),
#             'Total Correct: ' + str(np.round(totalCorrect/usableTrialTotal,2)),
#             'Rewards this session:  ' + str(len(df[df['resp']==1]))]
#            
#    if returnAs == 'str_array'.lower():
#            return array 
#        
#        
#    elif returnAs == 'num_array'.lower():   # returns array of ints and floats, no titles or words
#        
#         array = [d['wheelRewardDistance'][()],
#                  d['normRewardDistance'][()],
#                  d['maxResponseWaitFrames'][()],
#                  d['probGoRight'][()],
#                  np.round(sessionDuration/framerate/60, 2),
#                  len(repeats)/(totalTrials),
#                  ignored[1]+ignored[-1],
#                  usableTrialTotal,
#                  ignored[1],
#                  len(right),
#                  np.round(rightCorrect/len(right), 2),
#                  np.round(rightIncorrect/len(right),2),
#                  np.round(rightNoResp/len(right),2),
#                  ignored[-1],
#                  len(left),
#                  np.round(leftCorrect/len(left), 2),
#                  np.round(leftIncorrect/len(left),2),
#                  np.round(leftNoResp/len(left),2),
#                  np.round((leftCorrect+rightCorrect)/respTotal,2),
#                  np.round(totalCorrect/usableTrialTotal,2),
#                  len(df[df['resp']==1])]
#         
#         return array
#         
#    else:
#
#        if 'wheelRewardDistance' in d.keys():
#            print('Wheel Reward Dist: ' + str(d['wheelRewardDistance'][()]))
#        for a in array:
#            print(a)
#        
#        
#
#    
#    
