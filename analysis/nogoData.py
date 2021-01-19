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
from dataAnalysis import ignore_after

def nogo_turn(data, ignoreRepeats=True, ignoreNoRespAfter=None, returnArray=True):

    d = data
    end = ignore_after(d, ignoreNoRespAfter)[0] if ignoreNoRespAfter is not None else len(d['trialResponse'][:])
    
    trialResponse = d['trialResponse'][:end]
    trialTargetFrames = d['trialTargetFrames'][:end]
    trialMaskContrast = d['trialMaskContrast'][:end]
    trialStimStart = d['trialStimStartFrame'][:end]
    trialRespFrame = d['trialResponseFrame'][:end]
    trialRewardDir = d['trialRewardDir'][:end]
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
        trialRewardDir = trialRewardDir[prevTrialIncorrect==False]
    
    wheelMvmt = [[],[]]     # first is nogo, 2nd maskOnly
    ind = [[],[]]           # indices of above trials

# determines direction mouse turned during trials with no target

    for i, (start, end, resp, target, rew, mask) in enumerate(
            zip(trialStimStart, trialRespFrame, trialResponse, 
                trialTargetFrames, trialRewardDir, trialMaskContrast)):
        if target==0 and rew==0 and resp==-1:                      ### will mask Only have rewdir of 0 or nan?
            wheel = (np.cumsum(deltaWheel[start:end])[-1])
            if mask==0:             # nogos
                ind[0].append(i)
                wheelMvmt[0].append((wheel/abs(wheel)).astype(int))   
            elif mask>0:            # maskOnly
                ind[1].append(i)
                wheelMvmt[1].append((wheel/abs(wheel)).astype(int))

# returns array of turning directions [-1,1, etc] for trial types and indices of those trials
    if returnArray==True:    
        return [wheelMvmt, ind]

#catchTrials = [i for i, row in df.iterrows() if row.isnull().any()]


#
#    trials = d['trialType'][:end]
#    
#    for i, (start, end, t) in enumerate(zip(trialStimStart, trialRespFrame, trials)):
#        if t == 'maskOnly':
#            wheel = (np.cumsum(deltaWheel[start:end])[-1])
#            ind[1].append(i)
#            wheelMvmt[1].append((wheel/abs(wheel)).astype(int))
#
#
#
#



















