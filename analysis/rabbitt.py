# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:49:08 2020

@author: chelsea.strawder

Creates a plot of the outcome of next correct trial following either an incorrect or correct trial,
and the outcome time of that trial

Based on "Rabbitt" effect - slower mean reaction times after an error in humans 
(error-correction)

Next steps are to add a plot that separates response by side, especially for mice with strong biases

"""

from dataAnalysis import get_dates
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
sns.set_style('white')


def rabbitt(dataframe):
    
    df = dataframe
    currentTrialPrevError = []  # inds of current trial
    currentTrialPrevCorr = []
    
    trialThatIsPrevError = []  # these are the inds of the prev trial to above trials
    trialThatIsPrevCorr = []
    
    
    # get indices of trials where prev trial was either corr/error 
    # e is the ind of previous trial, e+1 is ind of current trial (That we want the time for)
    for e, (prev, current) in enumerate(zip(df['resp'][:-1], df['resp'][1:])):
        
        if current==1:
            if (df.loc[e, 'rewDir'] !=0) and (df.loc[e+1, 'rewDir'] !=0):
                if (df.loc[e, 'ignoreTrial'] !=True) and (df.loc[e+1, 'ignoreTrial'] !=True):
                    if (df.loc[e, 'outcomeTime_ms'] >0) and (df.loc[e+1, 'outcomeTime_ms']>0):
                        if prev==-1:
                            currentTrialPrevError.append(e+1)
                            trialThatIsPrevError.append(e)
                        elif prev==1:
                                currentTrialPrevCorr.append(e+1)  # append the ind of the current trial
                                trialThatIsPrevCorr.append(e)
                        else:
                            pass
    
    # use indices to get outcome times from df    
    # these are the times for the previous trial
    timePrevIncorrect = [df.loc[time, 'outcomeTime_ms'] for time in trialThatIsPrevError]
    timePrevCorrect = [df.loc[time, 'outcomeTime_ms'] for time in trialThatIsPrevCorr]
    
    #these are all correct, current trials; preceding trials differ (above)
    timeCurrTrialWithPrevIncorrect = [df.loc[time, 'outcomeTime_ms'] for time in currentTrialPrevError]
    timeCurrTrialWithPrevCorrect = [df.loc[time, 'outcomeTime_ms'] for time in currentTrialPrevCorr]
    
    
    date = get_dates(df)
    
    plt.figure()
    sns.barplot(data=[timeCurrTrialWithPrevIncorrect, timeCurrTrialWithPrevCorrect]) # plots the mean
    plt.ylabel('Response Time of Next Trial (correct), (ms)')
    plt.title('Average Response Time Current vs Previous Trial')
    plt.suptitle(df.mouse + '  ' +  date)
    plt.xlabel('Previous Trial Type')
    plt.xticks(ticks=[0,1], labels=['incorrect', 'correct'])
    
    axisMax = df['outcomeTime_ms'].max()
    axisMin = df['outcomeTime_ms'].min()
    
    fig, ax = plt.subplots()
    plt.scatter(timePrevIncorrect, timeCurrTrialWithPrevIncorrect, color='m', alpha=.5)
    plt.title('Response Time of Current Trial vs Previous Incorrect Trial')
    plt.suptitle(df.mouse + '  ' + df.date)
    plt.xlabel('Response Time, Previous Trial Incorrect (ms))')
    plt.ylabel('Response Time, Current Trial Correct (ms))')
    ax.plot((axisMin,axisMax), (axisMin,axisMax), ls="--", color='k', alpha=.3)
    ax.set_xlim(axisMin,axisMax)
    ax.set_ylim(axisMin,axisMax)
    
    
    fig, ax = plt.subplots()
    plt.scatter(timePrevCorrect, timeCurrTrialWithPrevCorrect, color='c', alpha=.5)
    plt.title('Response Time of Current Trial vs Previous Correct Trial')
    plt.suptitle(df.mouse + '  ' + df.date)
    plt.xlabel('Response Time, Previous Trial Correct (ms))')
    plt.ylabel('Response Time, Current Trial Correct (ms))')
    ax.plot((axisMin,axisMax),(axisMin,axisMax), ls="--", color='k', alpha=.3)
    ax.set_xlim(axisMin,axisMax)
    ax.set_ylim(axisMin,axisMax)