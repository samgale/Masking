# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:49:08 2020

@author: chelsea.strawder
"""

from dataAnalysis import import_data, create_df, get_dates
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
sns.set_style('white')

d = import_data()
df = create_df(d)


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
                if prev==-1:
                    currentTrialPrevError.append(e+1)
                    trialThatIsPrevError.append(e)
                elif prev==1:
                        currentTrialPrevCorr.append(e+1)  # append the ind of the current trial
                        trialThatIsPrevCorr.append(e)

# use indices to get outcome times from df    
# these are the times for the previous trial
timePrevIncorrect = [df.loc[time, 'outcomeTime'] for time in trialThatIsPrevError]
timePrevCorrect = [df.loc[time, 'outcomeTime'] for time in trialThatIsPrevCorr]

#these are all correct, current trials; preceding trials differ (above)
timeCurrTrialWithPrevIncorrect = [df.loc[time, 'outcomeTime'] for time in currentTrialPrevError]
timeCurrTrialWithPrevCorrect = [df.loc[time, 'outcomeTime'] for time in currentTrialPrevCorr]


date = get_dates(df)

plt.figure()
sns.barplot(data=[timeCurrTrialWithPrevIncorrect, timeCurrTrialWithPrevCorrect])

plt.ylabel('Response Time of Next Trial (correct), (ms)')
plt.title('Response Time vs Previous Trial')
plt.suptitle(df.mouse + '  ' +  date)
plt.xlabel('Previous Trial Type')
plt.xticks(ticks=[0,1], labels=['incorrect', 'correct'])


fig, ax = plt.subplots()
plt.scatter(timePrevIncorrect, timeCurrTrialWithPrevIncorrect, color='m', alpha=.5)
plt.title('Response Time of Current Trial vs Previous Incorrect Trial')
plt.suptitle(df.mouse + '  ' + df.date)
plt.xlabel('Response Time, Previous Trial Incorrect (ms))')
plt.ylabel('Response Time, Current Trial Correct (ms))')
ax.plot((100,800), (100,800), ls="--", color='k', alpha=.3)
ax.set_xlim(100,800)
ax.set_ylim(100,800)


fig, ax = plt.subplots()
plt.scatter(timePrevCorrect, timeCurrTrialWithPrevCorrect, color='c', alpha=.5)
plt.title('Response Time of Current Trial vs Previous Correct Trial')
plt.suptitle(df.mouse + '  ' + df.date)
plt.xlabel('Response Time, Previous Trial Correct (ms))')
plt.ylabel('Response Time, Current Trial Correct (ms))')
ax.plot((100,800),(100,800), ls="--", color='k', alpha=.3)
ax.set_xlim(100,800)
ax.set_ylim(100,800)