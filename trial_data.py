# -*- coding: utf-8 -*-
"""
Created on Wed May 01 10:47:39 2019

@author: svc_ccg
"""
"""
This organizes data from the mouse files we want to analyze, orders them by date, and adds them to a dataframe.
It then plots the columns of the data frame over time, 1 plot for each mouse.  You can add more columns to analyze by
using multiindexing and adding a column variable.
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
    
              
def trials(data):
    trialResponse = d['trialResponse'].value
    answered_trials = np.count_nonzero(trialResponse)
    correct = (trialResponse==1).sum()
    percentCorrect = (correct/float(answered_trials)) * 100
    return percentCorrect




mouseID = []
expDate = []
percentCorrect = []

mice = ['439508', '439506', '439502', '441357', '441358']


for mouse in mice:
    files,dates = get_files(mouse)
    print(mouse + '=============')
    for i, (f,date) in enumerate(zip(files,dates)):
        d = h5py.File(f)
        #df.append((trials(d)))
        print(trials(d))
        mouseID.append(mouse)
        expDate.append(date)
        percentCorrect.append(trials(d))
        

rows = pd.MultiIndex.from_arrays([mouseID,expDate],names=('mouse','date'))   
df = pd.DataFrame(index=rows)  
df['percentCorrect'] = percentCorrect

for m in mice:
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    dt = df.loc[m].index
    dt -= dt[0]
    days = dt.days
    ax.plot(days,df.loc[m]['percentCorrect']/100,'-ko')
    ax.plot([0,max(days)],[0.5]*2,'k--')
    ax.set_xlim([-0.5,max(days)+0.5])
    ax.set_ylim([0,1])
    ax.set_title(m)
      
