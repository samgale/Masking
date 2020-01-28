# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:47:02 2019

@author: chelsea.strawder


I want this to also return the index of the nogos/maskOnly

"""

import h5py
import fileIO
import numpy as np
from ignoreTrials import ignore_trials


def nogo_turn(data, ignoreRepeats=True, returnArray=True):

    d = data
    trialResponse = d['trialResponse'][:]
    trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]
    trialMaskContrast = d['trialMaskContrast'][:len(trialResponse)]

    if 0 not in trialTargetFrames:
        print('There were no nogo trials')
    else:

        trialRespFrames = d['trialResponseFrame'][:]
        trialOpenLoop = d['trialOpenLoopFrames'][:len(trialResponse)] 
        trialStimStart = d['trialStimStartFrame'][:len(trialResponse)]
        deltaWheel = d['deltaWheelPos'][:]
        repeats = d['incorrectTrialRepeats'][()]
        timeout = d['incorrectTimeoutFrames'][()]
        preStim = d['preStimFramesFixed'][()]
   
        if ignoreRepeats == True and repeats>0: 
            trialResponseOG = d['trialResponse'][:]
            if 'trialRepeat' in d.keys():
                prevTrialIncorrect = d['trialRepeat'][:len(trialResponseOG)]
            else:
                prevTrialIncorrect = np.concatenate(([False],trialResponseOG[:-1]<1))
            
            trialResponse = trialResponseOG[prevTrialIncorrect==False]
            trialTargetFrames = trialTargetFrames[prevTrialIncorrect==False]
            trialStimStart = stimStart[(prevTrialIncorrect==False)]
            trialRespFrames = trialRespFrames[prevTrialIncorrect==False]
            trialOpenLoop = trialOpenLoop[prevTrialIncorrect==False]
            trialMaskContrast = trialMaskContrast[prevTrialIncorrect==False]
            
        elif ignoreRepeats==False:
            trialResponse = d['trialResponse'][:]

        deltaWheel = d['deltaWheelPos'][:]    
        
        startWheelPos = [[],[]]  # first is nogo, 2nd maskOnly
        endWheelPos = [[],[]]
        ind = [[],[]]   # index of trials

        ## take nogo wheel trace from the start of the closed loop
        ## extend wheel trace into the timeout period - in masked session no timeout, so use next trial prestim period

        for i, (start, end, resp, target, mask) in enumerate(
                zip(trialStimStart, trialRespFrames, trialResponse, trialTargetFrames, trialMaskContrast)):
            if mask==0:
                if target==0 and resp==-1:  # nogos
                      endWheelPos[0].append(deltaWheel[end+30])
                      startWheelPos[0].append(deltaWheel[start])
                      ind[0].append(i)
            elif mask>0 and target==0 and resp==-1:   # maskOnly
                endWheelPos[1].append(deltaWheel[end+30])
                startWheelPos[1].append(deltaWheel[start])
                ind[1].append(i)
        
        nogoWheel = (np.array(endWheelPos[0])) - (np.array(startWheelPos[0]))
        maskOnlyWheel = (np.array(endWheelPos[1])) - (np.array(startWheelPos[1]))
        
        nogoTurnDir = []   #returns an array of values that show the direction turned for ALL no-go trials,
        maskOnlyTurnDir = []
        
        for i in nogoWheel:
            if i >0:
                nogoTurnDir.append(1)
            else:
                nogoTurnDir.append(-1)
        
        for f in maskOnlyWheel:
            if f>0:
                maskOnlyTurnDir.append(1)
            else:
                maskOnlyTurnDir.append(-1)
        
        nogoTurnDir = np.array(nogoTurnDir) 
        maskOnlyTurnDir = np.array(maskOnlyTurnDir)
        
    if returnArray==True:    
        return [nogoTurnDir, maskOnlyTurnDir, ind]
