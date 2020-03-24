# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:47:02 2019

@author: chelsea.strawder

This returns 3 arrays - 
1st has turning direction for incorrect nogo trials 
2nd has turning direction for incorrect maskOnly trials (during masking)
3rd has 2 arrays - the indices of the above 

"""

import numpy as np


def nogo_turn(data, ignoreRepeats=True, returnArray=True):

    d = data
    trialResponse = d['trialResponse'][:]
    end = len(trialResponse)
    trialTargetFrames = d['trialTargetFrames'][:end]
    trialMaskContrast = d['trialMaskContrast'][:end]
    trialStimStart = d['trialStimStartFrame'][:end]
    trialRespFrame = d['trialResponseFrame'][:]
    deltaWheel = d['deltaWheelPos'][:]
    repeats = d['incorrectTrialRepeats'][()]
   
    if ignoreRepeats == True and repeats>0: 
        if 'trialRepeat' in d.keys():
            prevTrialIncorrect = d['trialRepeat'][:end]
        else:
            prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))
        
        trialResponse = trialResponse[prevTrialIncorrect==False]
        trialTargetFrames = trialTargetFrames[prevTrialIncorrect==False]
        trialStimStart = trialStimStart[(prevTrialIncorrect==False)]
        trialRespFrame = trialRespFrame[prevTrialIncorrect==False]
        trialMaskContrast = trialMaskContrast[prevTrialIncorrect==False]  
    
    wheelMvmt = [[],[]]     # first is nogo, 2nd maskOnly
    ind = [[],[]]           # indices of above trials

    # determines direction mouse turned during trials with no target

    for i, (start, end, resp, target, mask) in enumerate(
            zip(trialStimStart, trialRespFrame, trialResponse, trialTargetFrames, trialMaskContrast)):
        if target==0 and resp==-1:
            wheel = (np.cumsum(deltaWheel[start:end])[-1])
            if mask==0:             # nogos
                ind[0].append(i)
                wheelMvmt[0].append((wheel/abs(wheel)).astype(int))   
            elif mask>0:            # maskOnly
                ind[1].append(i)
                wheelMvmt[1].append((wheel/abs(wheel)).astype(int))

    if returnArray==True:    
        return [wheelMvmt, ind]


