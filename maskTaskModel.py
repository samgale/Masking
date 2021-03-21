# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import pickle
import random
import numpy as np
import scipy.optimize
import scipy.signal
import scipy.stats
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from numba import njit
import fileIO



def fitModel(fitParamRanges,fixedParams):
    fit = scipy.optimize.brute(calcModelError,fitParamRanges,args=fixedParams,full_output=True,finish=None)
    #finalFit = scipy.optimize.minimize(getModelError,fit[0],args=(target,mask,maskOnset,optoOnset,responseRate,fractionCorrect),method='Nelder-Mead')
    return fit[0]


def calcModelError(paramsToFit,*fixedParams):
    sigma,decay,inhib,threshold,trialEnd = paramsToFit
    signals,targetSide,maskOnset,optoOnset,trialsPerCondition,responseRate,fractionCorrect = fixedParams
    trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition)
    result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)
    respRateError = np.nansum((responseRate-result['responseRate'])**2)
    fracCorrError = np.nansum((fractionCorrect-result['fractionCorrect'])**2)
    return respRateError + fracCorrError


def analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime):
    result = {}
    responseRate = []
    fractionCorrect = []
    for side in targetSide:
        result[side] = {}
        sideTrials = trialTargetSide==side
        mo = [np.nan] if side==0 else maskOnset
        for maskOn in mo:
            result[side][maskOn] = {}
            maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
            for optoOn in optoOnset:
                optoTrials = np.isnan(trialOptoOnset) if np.isnan(optoOn) else trialOptoOnset==optoOn
                trials = sideTrials & maskTrials & optoTrials
                responded = response[trials]!=0
                responseRate.append(np.sum(responded)/np.sum(trials))
                result[side][maskOn][optoOn] = {}
                result[side][maskOn][optoOn]['responseRate'] = responseRate[-1]
                result[side][maskOn][optoOn]['responseTime'] = responseTime[trials][responded]
                if side!=0 and maskOn!=0:
                    correct = response[trials]==side
                    fractionCorrect.append(np.sum(correct[responded])/np.sum(responded))
                    result[side][maskOn][optoOn]['fractionCorrect'] = fractionCorrect[-1]
                else:
                    fractionCorrect.append(np.nan)
    result['responseRate'] = np.array(responseRate)
    result['fractionCorrect'] = np.array(fractionCorrect)
    return result


def runSession(signals,targetSide,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition,record=False):
    trialTargetSide = []
    trialMaskOnset = []
    trialOptoOnset = []
    response = []
    responseTime = []
    Lrecord = []
    Rrecord = []
    Linitial = Rinitial = 0
    for side in targetSide:
        mo = [np.nan] if side==0 else maskOnset
        for maskOn in mo:
            if np.isnan(maskOn):
                sig = 'targetOnly'
                maskOn = np.nan
            elif maskOn==0:
                sig = 'maskOnly'
            else:
                sig = 'mask'
            for optoOn in optoOnset:
                if side==0:
                    Lsignal = np.zeros(signals[sig]['ipsi'][maskOn].size)
                    Rsignal = Lsignal.copy()
                elif side<0:
                    Lsignal = signals[sig]['contra'][maskOn].copy()
                    Rsignal = signals[sig]['ipsi'][maskOn].copy()
                else:
                    Lsignal = signals[sig]['ipsi'][maskOn].copy()
                    Rsignal = signals[sig]['contra'][maskOn].copy()
                if not np.isnan(optoOn):
                    Lsignal[int(optoOn):] = 0
                    Rsignal[int(optoOn):] = 0
                for _ in range(trialsPerCondition):
                    trialTargetSide.append(side)
                    trialMaskOnset.append(maskOn)
                    trialOptoOnset.append(optoOn)
                    result = runTrial(sigma,decay,inhib,threshold,trialEnd,Linitial,Rinitial,Lsignal,Rsignal,record)
                    response.append(result[0])
                    responseTime.append(result[1])
                    if record:
                        Lrecord.append(result[2])
                        Rrecord.append(result[3])
    return np.array(trialTargetSide),np.array(trialMaskOnset),np.array(trialOptoOnset),np.array(response),np.array(responseTime),Lrecord,Rrecord


@njit
def runTrial(sigma,decay,inhib,threshold,trialEnd,Linitial,Rinitial,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(Lsignal.size,np.nan)
        Rrecord = Lrecord.copy()
    else:
        Lrecord = Rrecord = None
    L = Linitial
    R = Rinitial
    t = 0
    response = 0
    while t<trialEnd and response==0:
        L += random.gauss(0,sigma) + Lsignal[t] - decay*L - inhib*R 
        R += random.gauss(0,sigma) + Rsignal[t] - decay*R - inhib*L
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


def createSignals(psth,maskOnset,gamma):
    target = psth['targetOnly']['contra'][np.nan]
    mask = psth['maskOnly']['contra'][0]
    signals = {}
    for sig in psth.keys():
        signals[sig] = {}
        for hemi in ('ipsi','contra'):
            signals[sig][hemi] = {}
            if sig=='targetOnly':
                signals[sig][hemi][np.nan] = psth[sig][hemi][np.nan]
            elif sig=='maskOnly':
                signals[sig][hemi][0] = psth[sig][hemi][0]
            else:
                for mo in maskOnset[(~np.isnan(maskOnset)) & (maskOnset>0)]:
                    msk = np.zeros(mask.size)
                    msk[int(mo):] = mask[:-int(mo)]
#                    signals[sig][hemi][mo] = target+msk*(1-np.exp(-mo/tau)) if hemi=='contra' else msk
                    trg = target
                    trg[trg<0] = 0
                    signals[sig][hemi][mo] = target+msk/(1+gamma*trg) if hemi=='contra' else msk
    return signals


def normalizeSignals(signals):
    smax = max([signals[sig][hemi][mo].max() for sig in signals.keys() for hemi in ('ipsi','contra') for mo in signals[sig][hemi]])
    for sig in signals.keys():
        for hemi in ('ipsi','contra'):
            for mo in signals[sig][hemi]:
                signals[sig][hemi][mo] = signals[sig][hemi][mo]/smax
    return None


def calcSignalError(gamma,*params):
    psth,maskOnset = params
    signals = createSignals(psth,maskOnset,gamma)
    sse = []
    for sig in psth.keys():
        for hemi in ('ipsi','contra'):
            for mo in psth[sig][hemi]:
                sse.append(np.sum((signals[sig][hemi][mo]-psth[sig][hemi][mo])**2))
    return sum(sse)


def plotSignals(signalList,tmes,clrs):
    fig = plt.figure(figsize=(6,10))
    n = 2+len(signalList[-1]['mask']['contra'].keys())
    gs = matplotlib.gridspec.GridSpec(n,2)
    axs = []
    ymin = 0
    ymax = 0
    for j,hemi in enumerate(('ipsi','contra')):
        i = 0
        for sig in signalList[-1].keys():
            for mo in signalList[-1][sig][hemi]:
                ax = fig.add_subplot(gs[i,j])
                for d,t,clr in zip(signalList,tmes,clrs):
                    maskOn = list(d[sig][hemi].keys())[0] if np.isnan(mo) else mo
                    p = d[sig][hemi][maskOn]
                    ax.plot(t,p,clr)
                    ymin = min(ymin,p.min())
                    ymax = max(ymax,p.max())
                if i==n-1:
                    ax.set_xlabel('Time (ms)')
                else:
                    ax.set_xticklabels([])
                if j==0:
                    ax.set_ylabel('Spikes/s')
                    title = sig
                    if sig=='mask':
                        title += ', SOA '+str(round(mo/120*1000,1))+' ms'
                    title += ', '+hemi
                else:
                    ax.set_yticklabels([])
                    title = hemi
                ax.set_title(title)
                axs.append(ax)
                i += 1
    for ax in axs:
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,trialEndTimeMax])
        ax.set_ylim([1.05*ymin,1.05*ymax])
    plt.tight_layout()



# fixed parameters
dt = 1/120*1000
trialEndTimeMax = 200
trialEndMax = int(round(trialEndTimeMax/dt))
targetLatency = int(round(4/120*1000/dt))


# create model input signals from population ephys responses
popPsthFilePath = fileIO.getFile(fileType='*.pkl')
popPsth = pickle.load(open(popPsthFilePath,'rb'))

t = np.arange(0,trialEndMax*dt,dt)
signalNames = ('targetOnly','maskOnly','mask')

#filtPts = t.size
#expFilt = np.zeros(filtPts*2)
#expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=2,sym=False)
#expFilt /= expFilt.sum()

popPsthFilt = {}
for sig in signalNames:
    popPsthFilt[sig] = {}
    for hemi in ('ipsi','contra'):
        popPsthFilt[sig][hemi] = {}
        for mo in popPsth[sig][hemi]:
            p = np.interp(t,popPsth['t']*1000,popPsth[sig][hemi][mo])
#            p = np.interp(t,popPsth['t']*1000,scipy.signal.savgol_filter(popPsth[sig][hemi][mo],5,3))
#            p = np.interp(t,popPsth['t']*1000,np.convolve(popPsth[sig][hemi][mo],expFilt)[t.size:2*t.size])
            p -= p[t<=25].mean()
            maskOn = np.nan if sig=='targetOnly' else mo
            popPsthFilt[sig][hemi][maskOn] = p

plotSignals([popPsth,popPsthFilt],[popPsth['t']*1000,t],'kr')

normalizeSignals(popPsthFilt)


# sythetic signals based on ephys response to target and mask only
maskOnset = np.array([2,3,4,6])

gammaRange = slice(1,50,1)
fit = scipy.optimize.brute(calcSignalError,(gammaRange,),args=(popPsthFilt,maskOnset),full_output=True,finish=None)

gamma = 12
syntheticSignals = createSignals(popPsthFilt,maskOnset,gamma)

plotSignals([popPsthFilt,syntheticSignals],[t,t],'kr')



## fit model parameters
respRateFilePath = fileIO.getFile(fileType='*.npy')
respRateData = np.load(respRateFilePath)
respRateMean = np.nanmean(np.nanmean(respRateData,axis=1),axis=0)
respRateSem = np.nanmean(np.nanmean(respRateData,axis=1),axis=0)/(len(respRateData)**0.5)

fracCorrFilePath = fileIO.getFile(fileType='*.npy')
fracCorrData = np.load(fracCorrFilePath)
fracCorrMean = np.nanmean(np.nanmean(fracCorrData,axis=1),axis=0)
fracCorrSem = np.nanstd(np.nanmean(fracCorrData,axis=1),axis=0)/(len(fracCorrData)**0.5)

trialsPerCondition = 5000
targetSide = (1,0) # (-1,1,0)
maskOnset = [2,3,4,6,np.nan,0]
optoOnset = [np.nan]

sigmaRange = slice(0.05,0.55,0.05)
decayRange = slice(0,1,1) #slice(0,0.45,0.05)
inhibRange = slice(0,0.45,0.05)
thresholdRange = slice(0.5,5.5,0.5)
trialEndRange = slice(10,22,2)

signals = syntheticSignals

fitParamRanges = (sigmaRange,decayRange,inhibRange,thresholdRange,trialEndRange)
fixedParams = (signals,targetSide,maskOnset,optoOnset,trialsPerCondition,respRateMean,fracCorrMean)

fit = fitModel(fitParamRanges,fixedParams)

# out of sample fits
outOfSampleFits = []
nconditions = len(respRateMean)
for i in range(nconditions):
    if i==nconditions-1:
        tarSide = [ts for ts in targetSide if ts!=0]
        maskOn = maskOnset
        respRate = respRateMean[:-1]
        fracCorr = fracCorrMean[:-1]
    else:
        tarSide = targetSide
        maskOn = [mo for j,mo in enumerate(maskOnset) if j!=i]
        respRate,fracCorr = [np.array([d for j,d in enumerate(data) if j!=i]) for data in (respRateMean,fracCorrMean)]
    fixedParams=(signals,tarSide,maskOn,optoOnset,trialsPerCondition,respRate,fracCorr)
    outOfSampleFits.append(fitModel(fitParamRanges,fixedParams))
    

# plot fit results
sigma,decay,inhib,threshold,trialEnd = fit

trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=10000,record=True)

result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)

ylabel = {'responseRate': 'Response Rate',
          'fractionCorrect': 'Fraction Correct',
          'responseTime': 'Mean Decision Time (ms)'}


xticks = [mo*dt for mo in maskOnset[:-2]]+[67,83,100]
xticklabels = [str(int(round(x))) for x in xticks[:-3]]+['target\nonly','mask\nonly','no\nstimulus']
xlim = [8,108]
for measure,ylim,loc in  zip(('responseRate','fractionCorrect'),((0,1.05),(0.4,1.03)),('upper right','upper left')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    y = respRateMean if measure=='responseRate' else fracCorrMean
    d = result[measure]
    ax.plot(xticks,y,'o',mec='k',mfc='none',ms=8,mew=2,label='mice')
    ax.plot(xticks,d,'o',mec='r',mfc='none',ms=8,mew=2,label='model')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel[measure])
    ax.legend(loc=loc)


for side,lbl in zip((-1,1),('target left','target right')):
    sideTrials = targetSide==side
    for maskOn in maskOnset:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,200],[threshold,threshold],'k--')
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
        trial = np.where(sideTrials & maskTrials)[0][0]
        ax.plot(t,Rrecord[trial],'r')
        ax.plot(t,Lrecord[trial],'b')
        for axside in ('right','top','left','bottom'):
            ax.spines[axside].set_visible(False)
        ax.tick_params(direction='out',right=False,top=False,left=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([])
        ax.set_xlim([0,200])
        ax.set_ylim([-1.05*threshold,1.05*threshold])
        ax.set_xlabel('Time (ms)')
        title = lbl
        if not np.isnan(maskOn):
            title += ' + mask (' + str(int(round(maskOn*dt))) + ' ms)'
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
maskOnset = np.array([2,3,4,6,np.nan])
optoOnset = np.array([np.nan])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=5000)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


for measure,ylim in  zip(('responseRate','fractionCorrect','responseTime'),((0,1.05),(0.4,1.03),(80,140))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = []
    for maskOn in maskOnset:
        for optoOn in optoOnset:
            if measure=='responseTime':
                d.append(sum([result[side][maskOn][optoOn][measure].mean()*dt for side in (-1,1)])/2)
            else:
                d.append(sum([result[side][maskOn][optoOn][measure] for side in (-1,1)])/2)
    ax.plot(np.array(maskOnset[:-1])*dt,d[:-1],'ko')
    ax.plot(125,d[-1],'ko')
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
ax = fig.add_subplot(2,1,2)
clrs = plt.cm.plasma(np.linspace(0,1,len(maskOnset)))[::-1]
lbls = [str(round(mo*dt,1))+' ms' for mo in maskOnset[:-1]]+['target only']
xlim = [0,200]
ntrials = []
rt = []
for maskOn,clr in zip(maskOnset,clrs):
    trials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
    ntrials.append(trials.sum())
    respTrials = trials & (response!=0)
    rt.append(responseTime[respTrials].astype(float)*dt)
    c = (targetSide==response)[respTrials]
    p = []
    for i in t:
        j = (rt[-1]>=i) & (rt[-1]<i+dt)
        p.append(np.sum(c[j])/np.sum(j))
    ax.plot(t,p,'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([50,100,150,200])
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_xlabel('Model Decision Time (ms)')
ax.set_ylabel('Probability Correct')

ax = fig.add_subplot(2,1,1)
for r,n,clr,lbl in zip(rt,ntrials,clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([50,100,150,200])
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=8,loc='upper left')
plt.tight_layout()


# mask duration



# opto masking
maskOnset = np.array([2,np.nan])
optoOnset = np.array([0,2,4,6,8,10,12,14,16,np.nan])

targetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=5000)

result,maskOnset,optoOnset = analyzeSession(targetSide,trialMaskOnset,trialOptoOnset,response,responseTime)


x = np.array(optoOnset)*dt
x[-1] = x[-2] + 2*dt
for measure,ylim in  zip(('responseRate','fractionCorrect','responseTime'),((0,1.05),(0.4,1.03),(80,130))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)  
    if measure=='responseRate':
        side = 0
        d = [result[side][np.nan][optoOn][measure] for optoOn in optoOnset]
        ax.plot(x[:-1],d[:-1],'m-o')
        ax.plot(x[-1],d[-1],'mo',label='no stim')
    for maskOn,clr,lbl in zip(maskOnset,'bc',('target + mask','target only')):
        d = []
        for optoOn in optoOnset:
            if measure=='responseTime':
                d.append(sum([result[side][maskOn][optoOn][measure].mean()*dt for side in (-1,1)])/2)
            else:
                d.append(sum([result[side][maskOn][optoOn][measure] for side in (-1,1)])/2)
        ax.plot(x[:-1],d[:-1],clr+'-o')
        ax.plot(x[-1],d[-1],clr+'o',label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks([0,50,100,x[-1]])
    ax.set_xticklabels([0,50,100,'no opto'])
    ax.set_xlim([0,x[-1]+2*dt])
    ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel[measure])
    if measure=='responseRate':
        ax.legend()
    plt.tight_layout()




