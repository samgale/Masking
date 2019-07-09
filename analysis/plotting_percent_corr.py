# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 16:15:11 2019

@author: svc_ccg
"""
from __future__ import division
import os
import numpy as np
import h5py
import datetime 
import scipy.stats
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure

def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'training_to_analyze')
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]     

mouse = raw_input('Enter Mouse ID:   ')

files = get_files(mouse)

fig, ax = plt.subplots()  

plotPoints = []
chanceRates= []

for i,f in enumerate(files):
    d = h5py.File(f)
    trialResponse = d['trialResponse'].value
    trialRewardDirection = d['trialRewardDir'].value[:-1]
    
    if len(trialResponse) < 100:
        continue
    
    if len(trialResponse)!=len(trialRewardDirection):
        trialRewardDirection = d['trialRewardDir'].value
    
    def count(resp, direction):
        return len(trialResponse[(trialResponse==resp) & (trialRewardDirection==direction)])
    
    turnRightTotal, turnLeftTotal = sum(trialRewardDirection==1), sum(trialRewardDirection==-1)
    
    # count(response, reward direction) where -1 is turn left 
    rightTurnCorr, leftTurnCorr = count(1,1), count(1,-1)
    rightTurnIncorrect, leftTurnIncorrect = count(-1,1), count(-1,-1)
    rightNoResp, leftNoResp = count(0,1), count(0,-1)
    
    respTotal = (turnLeftTotal + turnRightTotal) - (rightNoResp + leftNoResp)
    
    print(f.split('_')[-3:-1])
    print('Trials: ' + (str(len(trialResponse))))
    
    d.close()

    for num, denom in zip([(leftTurnCorr+rightTurnCorr)], [respTotal]):
        if num/denom < 1:        
            plotPoints.append(num/denom) 
            chanceRates.append(np.array(scipy.stats.binom.interval(0.95, np.sum(denom), 0.5))/np.sum(denom))
        
chanceRates = np.array(chanceRates)
ax.fill_between(np.arange(len(chanceRates)), chanceRates[:, 0], chanceRates[:, 1], color='g', alpha=0.2)
           
ax.plot(range(len(plotPoints)), plotPoints, 'ko-')
ax.text(0, len(plotPoints), mouse)
ax.set_ylim([0,1.1])
ax.set_xticks(range(len(plotPoints)))
formatFigure(fig, ax, title='Percent Correct Over Sessions, ' + str(f.split('_')[-3:-1]), xLabel='Session Number', yLabel='Percent Correct')
 



