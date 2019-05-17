# -*- coding: utf-8 -*-
"""
Created on Wed May 01 10:47:39 2019

@author: svc_ccg
"""
"""
This organizes data from the mouse files we want to analyze, orders them by date, and adds them to a dataframe.
It then plots each of the columns of the data frame over time, 1 plot w subplots for each mouse.  You can add more columns to analyze by
using multiindexing and adding a column variable to the dict of column key/value pairs.
"""

from __future__ import division
import os
import numpy as np
import h5py
import datetime 
from matplotlib import pyplot as plt
import pandas as pd


# for each mouse
def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'files_to_analyze')
    files = os.listdir(dataDir)
    dates = [datetime.datetime.strptime(f.split('_')[2],'%Y%m%d') for f in files]
    files,dates = zip(*[z for z in sorted(zip(files,dates),key=lambda i: i[1])])
#    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files], dates  
    
              
def trials(data1):
    trialResponse = data1['trialResponse'].value
    answered_trials = np.count_nonzero(trialResponse)
    correct = (trialResponse==1).sum()
    percentCorrect = (correct/float(answered_trials)) * 100
    return percentCorrect


def average(data):
    frameRate = data['frameRate'].value
    trialEndFrames = data['trialEndFrame'].value
    nTrials = trialEndFrames.size
    trialStartFrames = data['trialStartFrame'][:nTrials]
    
    trialResponse = data['trialResponse'][:nTrials]
    correctTrials = trialResponse > 0
    incorrectTrials = trialResponse < 0
    
    if 'trialResponseFrame' in data.keys():
        trialResponseFrame = data['trialResponseFrame'][:nTrials]
        rewardFrames = trialResponseFrame[correctTrials]
        incorrectFrames = trialResponseFrame[incorrectTrials]
    else:
        postRewardTargetFrames = data['postRewardTargetFrames'] if 'postRewardTargetFrames' in data.keys() else 0
        incorrectTimeoutFrames = data['incorrectTimeoutFrames'] if 'incorrectTimeoutFrames' in data.keys() else 0
        rewardFrames = trialEndFrames[trialResponse>0] - postRewardTargetFrames
        incorrectFrames = trialEndFrames[trialResponse<0] - incorrectTimeoutFrames
        
    trialStimStartFrames = data['trialStimStartFrame'][:nTrials] if 'trialStimStartFrame' in data.keys() else trialStartFrames + data['preStimFrames']
    
    rewardRespTime = (rewardFrames - trialStimStartFrames[correctTrials])/frameRate
    
    incorrectRespTime = (incorrectFrames - trialStimStartFrames[incorrectTrials])/frameRate
    
    return np.median(rewardRespTime), np.median(incorrectRespTime)
        

mouseID = []
expDate = []

dict1 = {'percentCorrect': [], 'avg_correctRespTime': [], 'avg_incorrectRespTime': [], 'numRewards': []}   # column values

mice = ['439508', '439506', '439502', '441357', '441358']


for mouse in mice:
    files,dates = get_files(mouse)
    print(mouse + '=============')
    for i, (f,date) in enumerate(zip(files,dates)):
        d = h5py.File(f)
       # print(trials(d))
        mouseID.append(mouse)
        expDate.append(date)
        dict1['percentCorrect'].append(trials(d))
        rewardTime, incorrectTime = average(d)
        dict1['avg_correctRespTime'].append(rewardTime)
        dict1['avg_incorrectRespTime'].append(incorrectTime)
        dict1['numRewards'].append(np.sum(d['trialResponse'][:]==1))
        
        

rows = pd.MultiIndex.from_arrays([mouseID,expDate],names=('mouse','date'))   
df = pd.DataFrame(index=rows)  

for key,value in dict1.iteritems():             #unpacking the dict of columns values into the dataframe
    df[key] = value


for m in mice:                                  # creates subplots for each mouse
    fig = plt.figure(figsize=(8,10))
    for i,key in enumerate(df.columns):
        ax = plt.subplot(df.shape[1],1,i+1)
        dt = df.loc[m].index
        dt -= dt[0]
        days = dt.days
        ax.plot(days,df.loc[m][key],'-ko')
        ax.plot([0,max(days)],[0.5]*2,'k--')
        ax.set_xlim([-0.5,max(days)+0.5])
#        ax.set_ylim([0,1])
        ax.set_ylabel(key)
        if i==0:
            ax.set_title(m)
      
