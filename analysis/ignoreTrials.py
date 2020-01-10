# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:56:20 2019

@author: chelsea.strawder

Returns a list of trials indices where they moved the wheel before a temporal threshold
Use these indices to exclude those trials from analysis



"""

import numpy as np
import scipy.signal


def ignore_trials(d):
    trialResponse = d['trialResponse'][()]
    end = len(trialResponse)
    trialRewardDirection = d['trialRewardDir'][:end]
    trialTargetFrames = d['trialTargetFrames'][:end]
    trialStimStartFrame = d['trialStimStartFrame'][:]
    trialResponseFrame = d['trialResponseFrame'][:end]    #if they don't respond, then nothing is recorded here - this limits df to length of this variable
    deltaWheel = d['deltaWheelPos'][:]                      # has wheel movement for every frame of session

    
    
    for i, trial in enumerate(trialTargetFrames):  # this is needed for older files; nogos were randomly assigned a dir
        if trial==0:
            trialRewardDirection[i] = 0
    
    
    # length of time from start of stim to response (or no response) for each trial
    trialTimes = []   
    for i, (start, resp) in enumerate(zip(trialStimStartFrame, trialResponseFrame)):
            respTime = (deltaWheel[start:resp])
            trialTimes.append(respTime)
    
    #since deltawheel provides the difference in wheel mvmt from trial to trial
    #taking the cumulative sum gives the actual wheel mvmt and plots as a smooth curve
    cumRespTimes = []   
    for i, time in enumerate(trialTimes):
        time = np.cumsum(time)
        smoothed = scipy.signal.medfilt(time, kernel_size=5)
        cumRespTimes.append(smoothed)
    
    rxnTimes = []
    for i, times in enumerate(cumRespTimes):
        booleanMask = (abs(times[:])>10)
        val = np.argmax(booleanMask)
        rxnTimes.append(val)
               
    ignoreTrials = []
    for i, t in enumerate(rxnTimes):     # 15 frames = 125 ms 
        if 1<t<10:                                 # correct nogos have a rxn time of 0
            ignoreTrials.append(i)
    return ignoreTrials


