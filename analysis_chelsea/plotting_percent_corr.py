# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 16:15:11 2019

@author: svc_ccg
"""
import os
import numpy as np
import h5py
import datetime 
import scipy.stats
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure

def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
#   directory = r'\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\active_mice'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'training_to_analyze')
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]     


def plot_responses(mouse, respType, total=None):   #mosue id, 'correct' 'incorrect' 'none', total = None for only responses, or all for all trials 
    
    #mouse = raw_input('Enter Mouse ID:   ')
    respType='correct'
    mouse='521266'
    files = get_files(mouse, 'training_')
    total = 'all'
    
    fig, ax = plt.subplots()  
    
    plotPoints = []
    chanceRates= []
    
    for i,f in enumerate(files):
        d = h5py.File(f)
        trialResponse = d['trialResponse'][:]
        trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]
        trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]
        
#        if len(trialResponse) < 100:  #don't use files that have less than 100 trials 
#            continue
        
        if len(trialResponse)!=len(trialRewardDirection):  # 312 had these 2 arrays the same length
            trialRewardDirection = d['trialRewardDir'][:]
        
        def count(resp, direction):
            return len(trialResponse[(trialResponse==resp) & (trialRewardDirection==direction) & (trialTargetFrames!=0)])
        
        turnRightTotal = sum((trialRewardDirection==1) & (trialTargetFrames!=0))
        turnLeftTotal = sum((trialRewardDirection==-1) & (trialTargetFrames!=0))
        
        # count(response, reward direction) where -1 is turn left 
        rightTurnCorr, leftTurnCorr = count(1,1), count(1,-1)
        rightTurnIncorrect, leftTurnIncorrect = count(-1,1), count(-1,-1)
        rightNoResp, leftNoResp = count(0,1), count(0,-1)
        
        respTotal = (turnLeftTotal + turnRightTotal) - (rightNoResp + leftNoResp)        
        
        no_goTotal = len(trialTargetFrames[trialTargetFrames==0])
        no_goCorrect = len(trialResponse[(trialResponse==1) & (trialTargetFrames==0)])     
        
        respTotal = (turnLeftTotal + turnRightTotal) - (rightNoResp + leftNoResp)
        
        print(f.split('_')[-3:-1])
        print('Trials: ' + (str(len(trialResponse))))
        
    
        if respType == 'correct':
            response = leftTurnCorr+rightTurnCorr
        elif respType == 'incorrect':
            response = leftTurnIncorrect+rightTurnIncorrect
        elif respType == 'none':
            response = rightNoResp+leftNoResp
        
        
        if total=='all':
            total = turnRightTotal+turnLeftTotal
        else:
            total = respTotal
    
        for num, denom, noNum, noDenom in zip(response, total, no_goCorrect, no_goTotal):   # here is where function arguments are used 
            if num/denom < 1:        
                plotPoints.append(num/denom) 
                chanceRates.append(np.array(scipy.stats.binom.interval(0.95, np.sum(denom), 0.5))/np.sum(denom))
            plt.plot(noNum/noDenom, 'go')    
        d.close()   
         
    chanceRates = np.array(chanceRates)
    ax.fill_between(np.arange(len(chanceRates)), chanceRates[:, 0], chanceRates[:, 1], color='g', alpha=0.2)
               
    ax.plot(range(len(plotPoints)), plotPoints, 'ko-')
    ax.text(0, len(plotPoints), mouse)
    ax.set_ylim([0,1.1])
    ax.set_xticks(range(len(plotPoints)))
    formatFigure(fig, ax, title='Percent Correct Over Sessions, ' + mouse, xLabel='Session Number', yLabel='Percent Correct')
     
    
    

