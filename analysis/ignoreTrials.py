# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:56:20 2019

@author: chelsea.strawder

Returns a list of trials indices where they moved the wheel before a threshold
Use these indices to exclude those trials from analysis

"""

import numpy as np
import scipy.signal


def ignore_trials(d):
    trialResponse = d['trialResponse'][()]
    end = len(trialResponse)
    trialStimStartFrame = d['trialStimStartFrame'][:]
    trialResponseFrame = d['trialResponseFrame'][:end]    #if they don't respond, then nothing is recorded here - this limits df to length of this variable
    deltaWheel = d['deltaWheelPos'][:]                      # has wheel movement for every frame of session
    qThreshold = d['maxQuiescentNormMoveDist'][()]*d['monSizePix'][0] 
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))

    trialWheel = [deltaWheel[start:resp] for (start, resp) in zip(trialStimStartFrame, trialResponseFrame)]
    cumWheel = [np.cumsum(time) for time in trialWheel]
    
    ignoreTrials = []
   # interpWheel = []
    for i, times in enumerate(cumWheel):
        fp = times    
        xp = np.arange(0, len(fp))*1/framerate
        x = np.arange(0, xp[-1], .001)          #wheel mvmt each ms 
        interp = np.interp(x,xp,fp)
      #  interpWheel.append(interp)
        val = np.argmax(abs(interp)>(qThreshold*1.5))
        if 0<val<100:
            ignoreTrials.append(i)

    return ignoreTrials



