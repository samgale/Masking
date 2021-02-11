# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import random
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numba import njit



def scoreSimulation(paramsToFit,*fixedParams):
    sigma,threshold = paramsToFit
    responseRate,fractionCorrect = fixedParams
    trialType,response,responseTime,Lrecord,Rrecord = runSession(sigma,threshold)
    modelRespRate = []
    modelFracCorr = []
    for label in set(trialType):
        trials = [trial==label for trial in trialType]
        responded = response[trials] != 0
        if not any(responded):
            return 1000
        correct = response[trials]==-1 if 'Left' in label else response[trials]==1
        modelRespRate.append(np.sum(responded)/np.sum(trials))
        modelFracCorr.append(np.sum(correct[responded])/np.sum(responded))
        
    respRateError = np.sum((np.array(responseRate)-np.array(modelRespRate))**2)
    fracCorrError = np.sum((np.array(fractionCorrect)-np.array(modelFracCorr))**2)
    
    return respRateError + 2*fracCorrError


@njit
def runSession(sigma,threshold,record=False):
    ntrials = 1000
    trialEnd = 150
    targetLatency = 40
    maskLatency = targetLatency + 17
    trialTypeLabels = ('targetLeft','targetRight','targetLeftMask','targetRightMask')
    trialInd = 0
    trialType = []
    response = []
    responseTime = []
    Lrecord = []
    Rrecord = []
    for n in range(ntrials):
        trialType.append(trialTypeLabels[trialInd])
        Lsignal = np.zeros(trialEnd)
        Rsignal = np.zeros(trialEnd)
        targetSignal = Lsignal if 'Left' in trialType[-1] else Rsignal
        targetSignal[targetLatency:targetLatency+50] = 0.7
        if 'Mask' in trialType[-1]:
            Lsignal[maskLatency:] = 1
            Rsignal[maskLatency:] = 1
        Linitial = Rinitial = 0
        result = runTrial(trialEnd,sigma,threshold,Linitial,Rinitial,Lsignal,Rsignal,record)
        response.append(result[0])
        responseTime.append(result[1])
        if record:
            Lrecord.append(result[2])
            Rrecord.append(result[3])
        if trialInd==len(trialTypeLabels)-1:
            trialInd = 0
        else:
            trialInd += 1
    
    return trialType,np.array(response),np.array(responseTime),Lrecord,Rrecord


@njit
def runTrial(trialEnd,sigma,threshold,Linitial,Rinitial,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(trialEnd,np.nan)
        Rrecord = np.full(trialEnd,np.nan)
    else:
        Lrecord = Rrecord = None
    L = Linitial
    R = Rinitial
    i = 0
    response = 0
    while i<trialEnd and response==0:
        L += Lsignal[i] + random.gauss(0,sigma)
        R += Rsignal[i] + random.gauss(0,sigma)
        if record:
            Lrecord[i] = L
            Rrecord[i] = R
        if L > threshold and R > threshold:
            response = -1 if L > R else 1
        elif L > threshold:
            response = -1
        elif R > threshold:
            response = 1
        i += 1
    responseTime = i+1
    
    return response,responseTime,Lrecord,Rrecord



responseRate = [0.6,0.6,0.9,0.9]
accuracy = [0.9,0.9,0.6,0.6]

sigmaRange = slice(1,20,1)
thresholdRange = slice(10,200,10)


fit = scipy.optimize.brute(scoreSimulation,(sigmaRange,thresholdRange),args=(responseRate,accuracy),full_output=True,finish=None)

fit[3][fit[3]>999]=np.nan


plt.imshow(fit[3],cmap='gray',origin='lower')

sigma,threshold = fit[0]

sigma,threshold = 0.5,100

trialType,response,responseTime,Lrecord,Rrecord = runSession(sigma,threshold,record=True)



plt.plot(Lrecord[-100])



trial = 4

plt.figure()
plt.plot([0,150],[threshold,threshold],'k--')
plt.plot([0,150],[-threshold,-threshold],'k--')
plt.plot(Lrecord[trial],'b')
plt.plot(-Rrecord[trial],'r')





















