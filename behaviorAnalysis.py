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

def makeWheelPlot(dataFile, returnData=False, responseFilter=[-1,0,1], frameRate=120):
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

    #Clean up inputs if needed    
    #if response filter is an int, make it a list
    if type(responseFilter) is int:
        responseFilter = [responseFilter]
    frameRate = float(frameRate)
    
    
    d = h5py.File(dataFile)
    trialStartFrames = d['trialStartFrame'].value
    trialEndFrames = d['trialEndFrame'].value
    trialRewardDirection = d['trialRewardDir'].value
    trialResponse = d['trialResponse'].value
    deltaWheel = d['deltaWheelPos'].value
    preStimFrames = d['trialPreStimFrames'] if 'trialPreStimFrames' in d else np.array([d['preStimFrames'].value]*trialStartFrames.size)
    openLoopFrames = d['openLoopFrames'].value
    
    if 'rewardFrames' in d.keys():
        rewardFrames = d['rewardFrames'].value
    elif 'responseFrames' in d.keys():
        responseTrials = np.where(trialResponse!= 0)[0]
        rewardFrames = d['responseFrames'].value[trialResponse[responseTrials]>0]
    else:
        rewardFrames = d['trialResponseFrame'].value[trialResponse>0]
            
    preFrames = preStimFrames
    fig, ax = plt.subplots()
    
    # for rightTrials stim presented on L, turn right - viceversa for leftTrials
    rightTrials = []
    leftTrials = []
    for i, (trialStart, trialEnd, rewardDirection, resp) in enumerate(zip(trialStartFrames, trialEndFrames, trialRewardDirection, trialResponse)):
        if i>0 and i<len(trialStartFrames):
            if resp in responseFilter:
                #get wheel position trace for this trial!
                trialWheel = np.cumsum(deltaWheel[trialStart+preFrames[i]:trialEnd])   
                trialreward = np.where((rewardFrames>trialStart)&(rewardFrames<=trialEnd))[0]
                reward = rewardFrames[trialreward[0]]-trialStart-preFrames[i] if len(trialreward)>0 else None
                if rewardDirection>0:
                    ax.plot(np.arange(trialWheel.size)/frameRate, trialWheel, 'r', alpha=0.2)
                    if reward is not None:
                        ax.plot(reward/frameRate, trialWheel[reward], 'ro')
                    rightTrials.append(trialWheel)
                else:
                    ax.plot(np.arange(trialWheel.size)/frameRate, trialWheel, 'b', alpha=0.2)
                    if reward is not None:
                        ax.plot(reward/frameRate, trialWheel[reward], 'bo')
                    leftTrials.append(trialWheel)
            
    rightTrials = pd.DataFrame(rightTrials).fillna(np.nan).values
    leftTrials = pd.DataFrame(leftTrials).fillna(np.nan).values
    ax.plot(np.arange(rightTrials.shape[1])/frameRate, np.nanmean(rightTrials,0), 'r', linewidth=3)
    ax.plot(np.arange(leftTrials.shape[1])/frameRate, np.nanmean(leftTrials, 0), 'b', linewidth=3)
    ax.plot([openLoopFrames/frameRate, openLoopFrames/frameRate], ax.get_ylim(), 'k--')
    
    formatFigure(fig, ax, xLabel='Time from stimulus onset (s)', yLabel='Wheel Position (pix)')
    
    if returnData:
        return rightTrials, leftTrials

def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'files_to_analyze')
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