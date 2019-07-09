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

def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'training_to_analyze')
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]     

mouse = raw_input('Enter Mouse ID:   ')

files = get_files(mouse)

for i,f in enumerate(files):
    d = h5py.File(f)
    trialResponse = d['trialResponse'].value
    trialRewardDirection = d['trialRewardDir'].value[:-1]
    
    def percent(resp, direction):
        return len(trialResponse[(trialResponse==resp) & (trialRewardDirection==direction)])
    
    turnRightTotal, turnLeftTotal = sum(trialRewardDirection==1), sum(trialRewardDirection==-1)
    
    # percent(response, reward direction) where -1 is turn left 
    rightTurnCorr, leftTurnCorr = percent(1,1), percent(1,-1)
    rightTurnIncorrect, leftTurnIncorrect = percent(-1,1), percent(-1,-1)
    rightNoResp, leftNoResp = percent(0,1), percent(0,-1)
    
    respTotal = (turnLeftTotal + turnRightTotal) - (rightNoResp + leftNoResp)
    
    print(f.split('_')[-3:-1])
    print('Trials: ' + (str(len(trialResponse))))
    
    plt.figure()
    ax = plt.subplot()
    
    for i, (num, denom) in enumerate(zip([rightTurnCorr, rightTurnIncorrect, rightNoResp, leftTurnCorr, leftTurnIncorrect, leftNoResp, (leftTurnCorr+rightTurnCorr)], 
                                         [turnRightTotal, turnRightTotal, turnRightTotal, turnLeftTotal, turnLeftTotal, turnLeftTotal, respTotal])):
        if num/denom < 1.0: 
            ax.text(0, len(files), mouse)
            ax.set_ylim([0,1.1])
            ax.plot(i, (num/denom))
         
            ax.set_xlabel('Session number')
            ax.set_ylabel('Percent Correct')
            
                                     
                                                   


