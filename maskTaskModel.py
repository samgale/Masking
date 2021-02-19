# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import random
import numpy as np
import scipy.optimize
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from numba import njit
import fileIO



def getModelError(paramsToFit,*fixedParams):
    sigma,decayRate,threshold = paramsToFit
    target,mask,maskOnset,optoOnset,responseRate,fractionCorrect = fixedParams
    targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(target,mask,maskOnset,optoOnset,sigma,decayRate,threshold)
    result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)
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
        fracCorrError = np.sum((1*(np.array(fractionCorrect)-np.array(modelFractionCorrect)))**2)
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
def runSession(target,mask,maskOnset,optoOnset,sigma,decayRate,threshold,record=False):
    trialsPerCondition = 1000
    trialEnd = 200
    targetLatency = 40
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
                    if side < 0:
                        Lsignal = target
                        Rsignal = np.zeros(trialEnd)
                    else:
                        Lsignal = np.zeros(trialEnd)
                        Rsignal = target
                    if not np.isnan(maskOn):
                        m = np.zeros(trialEnd)
                        m[maskOn:] = mask[:-maskOn]
                        Lsignal = np.maximum(Lsignal,m)
                        Rsignal = np.maximum(Rsignal,m)
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
    t = 0
    response = 0
    while t<trialEnd and response==0:
        L += random.gauss(0,sigma) + Lsignal[t] - decayRate*L 
        R += random.gauss(0,sigma) + Rsignal[t] - decayRate*R
        if record:
            Lrecord[t] = L
            Rrecord[t] = R
        if L > threshold and R > threshold:
            response = -1 if L > R else 1
        elif L > threshold:
            response = -1
        elif R > threshold:
            response = 1
        t += 1
    responseTime = t-1
    
    return response,responseTime,Lrecord,Rrecord



# create model input signals from mean ephys responses
f = fileIO.getFile()
d = np.load(f)
i = (d['t']>=0) & (d['t']<0.2)
t = np.arange(200)
signals = ('target','mask','targetMask')
target,mask,targetMask = [np.interp(t,d['t'][i]*1000,scipy.signal.savgol_filter(d[s][i],5,3)) for s in signals]


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for s,r,clr in zip(signals,(target,mask,targetMask),'kbr'):
    ax.plot(d['t'][i]*1000,d[s][i],clr)
    ax.plot(t,r,clr+':')

target /= mask.max()
mask /= mask.max()


# fit model parameters
maskOnset = np.array([np.nan,17])
optoOnset = np.array([np.nan])
responseRate = [0.4,0.8,0.4,0.8]
fractionCorrect = [0.9,0.55,0.9,0.55]

sigmaRange = slice(0.2,0.65,0.05)
decayRateRange = slice(0.25,0.7,0.05)
thresholdRange = slice(1,10,1)


fit = scipy.optimize.brute(getModelError,(sigmaRange,decayRateRange,thresholdRange),args=(target,mask,maskOnset,optoOnset,responseRate,fractionCorrect),full_output=True,finish=None)

sigma,decayRate,threshold = fit[0]

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(target,mask,maskOnset,optoOnset,sigma,decayRate,threshold,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)

ylabel = {'responseRate': 'Response Rate',
          'fractionCorrect': 'Fraction Correct',
          'responseTime': 'Decision Time (ms)'}


x = (0,1)
for measure,ylim,loc in  zip(('responseRate','fractionCorrect'),((0,1.05),(0.45,1.05)),('upper left','upper right')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = []
    for maskOn in maskOnset:
        for optoOn in optoOnset:
            d.append(sum([result[side][maskOn][optoOn][measure] for side in (-1,1)])/2)
    y = responseRate if measure=='responseRate' else fractionCorrect
    ax.plot(x,y[:2],'o',mec='k',mfc='none',ms=8,mew=2,label="~mouse")
    ax.plot(x,d,'o',mec='r',mfc='none',ms=8,mew=2,label="model")
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('target','target + mask'))
    ax.set_xlim([-0.5,1.5])
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel[measure])
    ax.legend(loc=loc)


for side,lbl in zip((-1,1),('target left','target right')):
    sideTrials = targetSide==side
    for maskOn in maskOnset:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,200],[threshold,threshold],'r--')
        ax.plot([0,200],[-threshold,-threshold],'b--')
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
        trial = np.where(sideTrials & maskTrials)[0][0]
        ax.plot(Rrecord[trial],'r')
        ax.plot(-Lrecord[trial],'b')
        for axside in ('right','top','left','bottom'):
            ax.spines[axside].set_visible(False)
        ax.tick_params(direction='out',right=False,top=False,left=False)
        ax.set_xticks([0,200])
        ax.set_yticks([])
        ax.set_xlim([0,200])
        ax.set_ylim([-1.05*threshold,1.05*threshold])
        ax.set_xlabel('Time (ms)')
        title = lbl
        if not np.isnan(maskOn):
            title += ' + mask'
        title += ', decision = '
        if response[trial]==-1:
            title += 'left'
        elif response[trial]==1:
            title += 'right'
        else:
            title += 'none'
        ax.set_title(title)
        plt.tight_layout()



# masking
maskOnset = np.array([np.nan,17,33,50,67,83,100])
optoOnset = np.array([np.nan])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


for measure,ylim in  zip(('responseRate','fractionCorrect','responseTime'),((0,1.05),(0.45,1.05),(60,140))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = []
    for maskOn in maskOnset:
        for optoOn in optoOnset:
            if measure=='responseTime':
                d.append(sum([result[side][maskOn][optoOn][measure].mean() for side in (-1,1)])/2)
            else:
                d.append(sum([result[side][maskOn][optoOn][measure] for side in (-1,1)])/2)
    ax.plot(maskOnset[1:],d[1:],'ko')
    ax.plot(125,d[0],'ko')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks([0,25,50,75,100,125])
    ax.set_xticklabels([0,25,50,75,100,'target only'])
    ax.set_xlim([0,137.5])
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel[measure])
    plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(2,1,1)
mo = maskOnset[1:] + [np.nan]
moLabels = [str(int(m)) for m in mo[:-1]] + ['target only']
clrs = plt.cm.plasma(np.linspace(0,1,len(maskOnset)))[::-1]
binWidth = 5
bins = np.arange(50,200,binWidth)
xlim = [50,200]
rt = []
for maskOn,clr in zip(mo,clrs):
    trials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
    trials = trials & (response!=0)
    rt.append(responseTime[trials].astype(float))
    c = (targetSide==response)[trials]
    p = []
    for i in bins:
        j = (rt[-1]>i) & (rt[-1]<i+binWidth)
        p.append(np.sum(c[j])/np.sum(j))
    ax.plot(bins,p,'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([50,100,150,200])
ax.set_xlim(xlim)
ax.set_ylim([0,1.01])
ax.set_ylabel('Probability Correct')

ax = fig.add_subplot(2,1,2)
yticks = np.arange(len(rt))
violin = ax.violinplot(rt,positions=yticks,vert=False,showextrema=False)
for v,clr in zip(violin['bodies'],clrs):
    v.set_color(clr)
    v.set_alpha(1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([50,100,150,200])
ax.set_yticks(yticks)
ax.set_yticklabels(moLabels)
ax.set_xlim(xlim)
ax.set_ylim([-0.5,len(rt)-0.5])
ax.set_xlabel('Decision Time (ms)')
ax.set_ylabel('SOA (ms)')
plt.tight_layout()


# mask duration



# opto masking
maskOnset = np.array([np.nan,17])
optoOnset = np.array([np.nan,17,33,50,67,83,100])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(maskOnset,optoOnset,sigma,decayRate,threshold,record=True)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


for measure,ylim in  zip(('responseRate','fractionCorrect','responseTime'),((0,1.05),(0.45,1.05),(50,110))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)   
    for maskOn,clr,lbl in zip(maskOnset,'cb',('target only','target + mask')):
        d = []
        for optoOn in optoOnset:
            if measure=='responseTime':
                d.append(sum([result[side][maskOn][optoOn][measure].mean() for side in (-1,1)])/2)
            else:
                d.append(sum([result[side][maskOn][optoOn][measure] for side in (-1,1)])/2)
        ax.plot(optoOnset[1:],d[1:],clr+'o')
        ax.plot(125,d[0],clr+'o',label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks([0,50,100,125])
    ax.set_xticklabels([0,50,100,'no opto'])
    ax.set_xlim([0,137.5])
    ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel[measure])
    if measure=='responseRate':
        ax.legend()
    plt.tight_layout()









