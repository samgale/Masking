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



def getModelError(paramsToFit,*fixedParams):
    sigma,decayRate,threshold,signalSigma = paramsToFit
    maskOnset,optoOnset,responseRate,fractionCorrect = fixedParams
    targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,signalSigma)
    result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response)
    modelResponseRate = []
    modelFractionCorrect = []
    for side in (-1,1):
        for maskOn in maskOnset:
            for optoOn in optoOnset:
                modelResponseRate.append(result[side][maskOn][optoOn]['responseRate'])
                modelFractionCorrect.append(result[side][maskOn][optoOn]['fractionCorrect'])
    if any(r==0 for r in modelResponseRate):
        return 1000
    else:
        respRateError = np.sum((np.array(responseRate)-np.array(modelResponseRate))**2)
        fracCorrError = np.sum((2*(np.array(fractionCorrect)-np.array(modelFractionCorrect)))**2)
        return respRateError + fracCorrError


def analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime):
    result = {}
    maskOnset = getOnsetTimes(trialMaskOnset)
    optoOnset = getOnsetTimes(trialOptoOnset)
    for side in (-1,1):
        sideTrials = targetSide==side
        result[side] = {}
        for maskOn in maskOnset:
            maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
            result[side][maskOn] = {}
            for optoOn in optoOnset:
                optoTrials = np.isnan(trialOptoOnset) if np.isnan(optoOn) else trialOptoOnset==optoOn
                trials = sideTrials & maskTrials & optoTrials
                responded = response[trials]!=0
                correct = response[trials]==side
                result[side][maskOn][optoOn] = {}
                result[side][maskOn][optoOn]['responseRate'] = np.sum(responded)/np.sum(trials)
                result[side][maskOn][optoOn]['fractionCorrect'] = np.sum(correct[responded])/np.sum(responded)
                result[side][maskOn][optoOn]['responseTime'] = responseTime[trials][responded]
    return result,maskOnset,optoOnset


def getOnsetTimes(trialOnsets):
    onset = np.unique(trialOnsets)
    if any(np.isnan(onset)):
        onset = [np.nan] + list(onset[~np.isnan(onset)])
    return onset


@njit
def runSession(maskOnset,optoOnset,sigma,decayRate,threshold,signalSigma,record=False):
    trialsPerCondition = 1000
    trialEnd = 200
    targetLatency = 50
    targetRespDur = 50
    targetAmp = 1
    maskAmp = 1
    targetSide = []
    trialMaskOnset = []
    trialOptoOnset = []
    response = []
    responseTime = []
    Lrecord = []
    Rrecord = []
    for side in (-1,1):
        for maskOn in maskOnset:
            for optoOn in optoOnset:
                for _ in range(trialsPerCondition):
                    targetSide.append(side)
                    trialMaskOnset.append(maskOn)
                    trialOptoOnset.append(optoOn)
                    Lsignal = np.zeros(trialEnd)
                    Rsignal = np.zeros(trialEnd)
                    targetSignal = Lsignal if side<0 else Rsignal
                    respNoise = random.gauss(0,signalSigma)
                    targetSignal[targetLatency:targetLatency+targetRespDur] = targetAmp + targetAmp/maskAmp*respNoise
                    if not np.isnan(maskOn):
                        maskLatency = targetLatency + maskOn
                        Lsignal[maskLatency:] = maskAmp + respNoise
                        Rsignal[maskLatency:] = maskAmp + respNoise
                    if not np.isnan(optoOn):
                        optoLatency = targetLatency + optoOn
                        Lsignal[optoLatency:] = 0
                        Rsignal[optoLatency:] = 0
                    Linitial = Rinitial = 0
                    result = runTrial(trialEnd,sigma,decayRate,threshold,Linitial,Rinitial,Lsignal,Rsignal,record)
                    response.append(result[0])
                    responseTime.append(result[1])
                    if record:
                        Lrecord.append(result[2])
                        Rrecord.append(result[3])
    
    return np.array(targetSide),np.array(trialMaskOnset),np.array(trialOptoOnset),np.array(response),np.array(responseTime),Lrecord,Rrecord


@njit
def runTrial(trialEnd,sigma,decayRate,threshold,Linitial,Rinitial,Lsignal,Rsignal,record=False):
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
        L += random.gauss(0,sigma) + Lsignal[i] - decayRate*L 
        R += random.gauss(0,sigma) + Rsignal[i] - decayRate*R
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



# fit model parameters
maskOnset = np.array([np.nan,17])
optoOnset = np.array([np.nan])
responseRate = [0.5,1,0.5,1]
fractionCorrect = [1,0.5,1,0.5]

sigmaRange = slice(0.05,0.5,0.05)
decayRateRange = slice(0.05,0.45,0.05)
thresholdRange = slice(2,10,2)
signalSigmaRange = slice(0,0.06,0.02)


fit = scipy.optimize.brute(getModelError,(sigmaRange,decayRateRange,thresholdRange,signalSigmaRange),args=(maskOnset,optoOnset,responseRate,fractionCorrect),full_output=True,finish=None)

sigma,decayRate,threshold,signalSigma = fit[0]

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,signalSigma,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response)


x = np.arange(4)
for measure,ylim in  zip(('responseRate','fractionCorrect'),((0,1.05),(0.45,1.05))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = []
    for side in (-1,1):
        for maskOn in maskOnset:
            for optoOn in optoOnset:
                d.append(result[side][maskOn][optoOn][measure])
    y = responseRate if measure=='responseRate' else fractionCorrect
    ax.plot(x,y,'o',mec='k',mfc='none',label="~mouse")
    ax.plot(x,d,'o',mec='r',mfc='none',label="model")
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('target left','target left\n+ mask','target right','target right\n+ mask'))
    ax.set_xlim([-0.5,3.5])
    ax.set_ylim(ylim)
    ax.set_ylabel(measure)
    ax.legend()



trials = range(0,4)
for trial in trials:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,200],[threshold,threshold],'r--')
    ax.plot([0,200],[-threshold,-threshold],'b--')
    ax.plot(Rrecord[trial],'r')
    ax.plot(-Lrecord[trial],'b')
    for side in ('right','top','left','bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,top=False,left=False)
    ax.set_xticks([0,200])
    ax.set_yticks([])
    ax.set_xlim([0,200])
    ax.set_ylim([-1.05*threshold,1.05*threshold])
    ax.set_xlabel('Time (ms)')
    ax.set_title(trialType[trial]+' , response = '+str(response[trial]))



# masking
maskOnset = np.array([np.nan,17,33,50,67,83,100])
optoOnset = np.array([np.nan])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,signalSigma,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


for measure,ylim in  zip(('responseRate','fractionCorrect','responseTime'),((0,1.05),(0.45,1.05),(50,150))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for side,clr in zip((-1,1),'br'):
        d = []
        for maskOn in maskOnset:
            for optoOn in optoOnset:
                d.append(result[side][maskOn][optoOn][measure])
        if measure=='responseTime':
            d = [r.mean() for r in d]
        ax.plot(maskOnset[1:],d[1:],clr+'o')
        ax.plot(150,d[0],clr+'o')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks([0,50,100,150])
    ax.set_xticklabels([0,50,100,'no mask'])
    ax.set_xlim([0,157.5])
    ax.set_ylim(ylim)
    ax.set_xlabel('SOA (ms)')
    ax.set_ylabel(measure)



# opto masking
maskOnset = np.array([np.nan,17])
optoOnset = np.array([np.nan,17,33,50,67,83,100])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,signalSigma,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


















