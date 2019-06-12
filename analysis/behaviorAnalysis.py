# -*- coding: utf-8 -*-
"""
Created on Fri May 03 18:44:40 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import h5py, os
from matplotlib import pyplot as plt
import pandas as pd 
import datetime
import matplotlib as mpl

mpl.rcParams['pdf.fonttype']=42

'''
    Makes plot of trial by trial wheel trajectory color coded by initial target position  
    INPUTS:
    dataFile: path to h5 file created at end of behavioral session
    returnData: whether to return the trial wheel data for go right and go left trials. Default is to not return data.
    responseFilter: list of trial responses to include. For example [-1, 1] to analyze only active response trials.
                    [1] to analyze only hits etc. Defaults to include all trials
    
    OUTPUTS:
    rightTrials, leftTrials: numpy arrays of all go right and go left trial wheel trajectories. Trials are padded
                            with nans to correct for variable length. 
'''
    
def makeWheelPlot(data, returnData=False, responseFilter=[-1,0,1], framesToShowBeforeStart=0):

    #Clean up inputs if needed    
    #if response filter is an int, make it a list
    if type(responseFilter) is int:
        responseFilter = [responseFilter]
    
    d = data
    frameRate = d['frameRate'].value
    trialEndFrames = d['trialEndFrame'][:]
    trialStartFrames = d['trialStartFrame'][:trialEndFrames.size]
    trialRewardDirection = d['trialRewardDir'][:trialEndFrames.size]
    trialResponse = d['trialResponse'][:trialEndFrames.size]
    deltaWheel = d['deltaWheelPos'].value
    preStimFrames = d['trialStimStartFrame'][:trialEndFrames.size]-trialStartFrames if 'trialStimStartFrame' in d else np.array([d['preStimFrames'].value]*trialStartFrames.size)
    openLoopFrames = d['openLoopFrames'].value    
    trialStartFrames += preStimFrames
    
    if 'rewardFrames' in d.keys():
        rewardFrames = d['rewardFrames'].value
    elif 'responseFrames' in d.keys():
        responseTrials = np.where(trialResponse!= 0)[0]
        rewardFrames = d['responseFrames'].value[trialResponse[responseTrials]>0]
    else:
        rewardFrames = d['trialResponseFrame'].value[trialResponse>0]

    fig, ax = plt.subplots()
    
    # for rightTrials stim presented on R, turn left - viceversa for leftTrials
    rightTrials = []
    leftTrials = []
    trialTime = (np.arange(max(trialEndFrames-trialStartFrames+framesToShowBeforeStart))-framesToShowBeforeStart)/frameRate  # evenly-spaced array of times for x-axis
    for i, (trialStart, trialEnd, rewardDirection, resp) in enumerate(zip(trialStartFrames, trialEndFrames, trialRewardDirection, trialResponse)):
        if i>0 and i<len(trialStartFrames):
            if resp in responseFilter:
                #get wheel position trace for this trial!
                trialWheel = np.cumsum(deltaWheel[trialStart-framesToShowBeforeStart:trialEnd])
                trialWheel -= trialWheel[0]
                trialreward = np.where((rewardFrames>trialStart)&(rewardFrames<=trialEnd))[0]
                rewardFrame = rewardFrames[trialreward[0]]-trialStart+framesToShowBeforeStart if len(trialreward)>0 else None
                if rewardDirection<0:
                    ax.plot(trialTime[:trialWheel.size], trialWheel, 'r', alpha=0.2)
                    if rewardFrame is not None:
                        ax.plot(trialTime[rewardFrame], trialWheel[rewardFrame], 'ro')
                    rightTrials.append(trialWheel)
                else:
                    ax.plot(trialTime[:trialWheel.size], trialWheel, 'b', alpha=0.2)
                    if rewardFrame is not None:
                        ax.plot(trialTime[rewardFrame], trialWheel[rewardFrame], 'bo')
                    leftTrials.append(trialWheel)
            
    rightTrials = pd.DataFrame(rightTrials).fillna(np.nan).values
    leftTrials = pd.DataFrame(leftTrials).fillna(np.nan).values
    ax.plot(trialTime[:rightTrials.shape[1]], np.nanmean(rightTrials,0), 'r', linewidth=3)
    ax.plot(trialTime[:leftTrials.shape[1]], np.nanmean(leftTrials, 0), 'b', linewidth=3)
    ax.plot([trialTime[framesToShowBeforeStart+openLoopFrames]]*2, ax.get_ylim(), 'k--')
    
    name_date = str(data).split('_')    
    
    formatFigure(fig, ax, xLabel='Time from stimulus onset (s)', yLabel='Wheel Position (pix)', title=name_date[-3:-1])
    
    if returnData:
        return rightTrials, leftTrials

def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'training_to_analyze')   #training_ for pre-mask, masking_ for mask
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]  
    
              
def trials(data):
    trialResponse = data['trialResponse'].value
    trials = np.count_nonzero(trialResponse)
    correct = (trialResponse==1).sum()
    percentCorrect = (correct/float(trials)) * 100
    return percentCorrect

def formatFigure(fig, ax, title=None, xLabel=None, yLabel=None, xTickLabels=None, yTickLabels=None, blackBackground=False, saveName=None):
    fig.set_facecolor('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
        
    if blackBackground:
        ax.set_axis_bgcolor('k')
        ax.tick_params(labelcolor='w', color='w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        for side in ('left','bottom'):
            ax.spines[side].set_color('w')

        fig.set_facecolor('k')
        fig.patch.set_facecolor('k')
    if saveName is not None:
        fig.savefig(saveName, facecolor=fig.get_facecolor())
        
def performanceByParam(data, paramName, units=''):
    
    if not paramName in data.keys():
        print('No parameter named ' + paramName + ' in hdf5 file')
        return
    
    d =  data
    
    trialResponse = d['trialResponse'].value    
    numTrials = len(trialResponse)
    
    trialRewardDirection = d['trialRewardDir'].value[:numTrials]
    trialParam = d[paramName][:numTrials] 
    
    if any(np.isnan(trialParam)):    
        trialParam[np.isnan(trialParam)] = -1

    paramValues = np.unique(trialParam)
    
    hits = [[],[]]
    misses = [[], []]
    noResps = [[],[]]
    for i, direction in enumerate([-1,1]):
        directionResponses = [trialResponse[(trialRewardDirection==direction) & (trialParam == p)] for p in paramValues]
        hits[i].append([np.sum(drs==1) for drs in directionResponses])
        misses[i].append([np.sum(drs==-1) for drs in directionResponses])
        noResps[i].append([np.sum(drs==0) for drs in directionResponses])
    
    hits = np.squeeze(np.array(hits))
    misses = np.squeeze(np.array(misses))
    noResps = np.squeeze(np.array(noResps))
    totalTrials = hits+misses+noResps
    
    for num, denom, title in zip([hits, hits, noResps], [totalTrials, hits+misses, totalTrials], ['total hit rate', 'response hit rate', 'no response rate']):
        fig, ax = plt.subplots()
        ax.plot(paramValues, num[0]/denom[0], 'ro-')
        ax.plot(paramValues, num[1]/denom[1], 'bo-')
        ax.set_ylim([0,1.01])
        ax.set_xlim([0, paramValues[-1]*1.1])
        ax.set_xticks(paramValues)
        formatFigure(fig, ax, xLabel=paramName + '('+units+')', yLabel='percent trials', title=title)
    
        
        
    
    
    
    
    
    
    
    
    