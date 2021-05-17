# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import copy
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



def fitModel(fitParamRanges,fixedParams,finish=False):
    fit = scipy.optimize.brute(calcModelError,fitParamRanges,args=fixedParams,full_output=False,finish=None,workers=1)
    if finish:
        finishRanges = []
        for rng,val in zip(fitParamRanges,fit):
            if val in (rng.start,rng.stop):
                finishRanges.append(slice(val,val+1,1))
            else:
                oldStep = rng.step
                newStep = oldStep/4
                finishRanges.append(slice(val-oldStep+newStep,val+oldStep,newStep))
        fit = scipy.optimize.brute(calcModelError,finishRanges,args=fixedParams,full_output=False,finish=None,workers=1)
    return fit


def calcModelError(paramsToFit,*fixedParams):
    alpha,nu,sigma,decay,inhib,threshold,trialEnd = paramsToFit
    signals,targetSide,maskOnset,optoOnset,trialsPerCondition,responseRate,fractionCorrect = fixedParams
    trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,alpha,nu,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition)
    result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)
    respRateError = np.nansum((responseRate-result['responseRate'])**2)
    fracCorrError = np.nansum((2*(fractionCorrect-result['fractionCorrect']))**2)
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
                    result[side][maskOn][optoOn]['responseTimeCorrect'] = responseTime[trials][responded & correct]
                    result[side][maskOn][optoOn]['responseTimeIncorrect'] = responseTime[trials][responded & (~correct)]
                else:
                    fractionCorrect.append(np.nan)
    result['responseRate'] = np.array(responseRate)
    result['fractionCorrect'] = np.array(fractionCorrect)
    return result


def runSession(signals,targetSide,maskOnset,optoOnset,alpha,nu,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition,record=False):
    trialTargetSide = []
    trialMaskOnset = []
    trialOptoOnset = []
    response = []
    responseTime = []
    Lrecord = []
    Rrecord = []
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
                if alpha > 0:
                    for s in (Lsignal,Rsignal):
                        i = s > 0
                        s[i] = s[i]**nu / (alpha**nu + s[i]**nu)
                        s *= alpha**nu + 1
                for _ in range(trialsPerCondition):
                    trialTargetSide.append(side)
                    trialMaskOnset.append(maskOn)
                    trialOptoOnset.append(optoOn)
                    result = runTrial(sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record)
                    response.append(result[0])
                    responseTime.append(result[1])
                    if record:
                        Lrecord.append(result[2])
                        Rrecord.append(result[3])
    return np.array(trialTargetSide),np.array(trialMaskOnset),np.array(trialOptoOnset),np.array(response),np.array(responseTime),Lrecord,Rrecord


@njit
def runTrial(sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(Lsignal.size,np.nan)
        Rrecord = Lrecord.copy()
    else:
        Lrecord = Rrecord = None
    L = 0
    R = 0
    t = 0
    response = 0
    while t<trialEnd and response==0:
        Lprev,Rprev = L,R
        L += random.gauss(0,sigma) + Lsignal[t] - decay*Lprev - inhib*Rprev
        R += random.gauss(0,sigma) + Rsignal[t] - decay*Rprev - inhib*Lprev
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


# fixed parameters
dt = 1/120*1000
trialEndTimeMax = 200
trialEndMax = int(round(trialEndTimeMax/dt))


# create model input signals from population ephys responses
popPsthFilePath = fileIO.getFile(fileType='*.pkl')
popPsth = pickle.load(open(popPsthFilePath,'rb'))

t = np.arange(0,trialEndMax*dt,dt)
signalNames = ('targetOnly','maskOnly','mask')

filtPts = t.size
expFilt = np.zeros(filtPts*2)
expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=2,sym=False)
expFilt /= expFilt.sum()

popPsthFilt = {}
for sig in signalNames:
    popPsthFilt[sig] = {}
    for hemi in ('ipsi','contra'):
        popPsthFilt[sig][hemi] = {}
        for mo in popPsth[sig][hemi]:
            p = popPsth[sig][hemi][mo].copy()
            p -= p[:,popPsth['t']<0].mean(axis=1)[:,None]
            p = p.mean(axis=0)
            
            p = p[(popPsth['t']>0) & (popPsth['t']<0.2)]
#            p = np.interp(t,popPsth['t']*1000,p)
#            p = np.interp(t,popPsth['t']*1000,scipy.signal.savgol_filter(p,5,3))
#            p = np.interp(t,popPsth['t']*1000,np.convolve(p,expFilt,mode='same'))
            
            p -= p[t<30].mean()
            maskOn = np.nan if sig=='targetOnly' else mo
            popPsthFilt[sig][hemi][maskOn] = p
                
                
# normalize and plot signals
signals = copy.deepcopy(popPsthFilt)

smax = max([signals[sig][hemi][mo].max() for sig in signals.keys() for hemi in ('ipsi','contra') for mo in signals[sig][hemi]])
for sig in signals.keys():
    for hemi in ('ipsi','contra'):
        for mo in signals[sig][hemi]:
            s = signals[sig][hemi][mo]
            s /= smax
#            i = s > 0
#            s[i] = s[i]**nu / (alpha**nu + s[i]**nu)
#            s[i] *= alpha**nu + 1

fig = plt.figure(figsize=(4,10))
n = 2+len(signals['mask']['contra'].keys())
axs = []
ymin = 0
ymax = 0
i = 0
for sig in signals:
    for mo in signals[sig]['contra']:
        ax = fig.add_subplot(n,1,i+1)
        for hemi,clr in zip(('ipsi','contra','ipsi'),'br'):
            p = signals[sig][hemi][mo]
            ax.plot(t,p,clr)
            ymin = min(ymin,p.min())
            ymax = max(ymax,p.max())
        if i==n-1:
            ax.set_xlabel('Time (ms)')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Spikes/s')
        title = sig
        if sig=='mask':
            title += ', SOA '+str(round(mo/120*1000,1))+' ms'
        title += ', '+hemi
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


# plot collapsing boundary
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#x = np.arange(t.size)
#times = np.arange(2,24,2)  
#for decayTiming,clr in zip(times,plt.cm.plasma(np.linspace(0,1,len(times)))):
#    for decayShape in [3]:
#        thresh = 1 - (1 - np.exp(-(x/decayTiming)**decayShape)) * (1*1)
#        ax.plot(t,thresh,color=clr,label='decayTiming='+str(int(decayTiming*dt))+', decayShape='+str(decayShape))
#ax.legend(fontsize=8)
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#shapes = np.arange(0.5,6.5,0.5)  
#for decayTiming in [12]:
#    for decayShape,clr in zip(shapes,plt.cm.plasma(np.linspace(0,1,len(shapes)))):
#        thresh = 1 - (1 - np.exp(-(x/decayTiming)**decayShape)) * (1*1)
#        ax.plot(t,thresh,color=clr,label='decayTiming='+str(int(decayTiming*dt))+', decayShape='+str(decayShape))
#ax.legend(fontsize=8)
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#initVals,times,shapes,amps = [np.arange(s.start,s.stop,s.step) for s in (thresholdRange,threshDecayTimingRange,threshDecayShapeRange,threshDecayAmpRange)] 
#for initVal in initVals:
#    for decayTiming in times:
#        for decayShape in shapes:
#            for decayAmp in amps:
#                thresh = initVal - (1 - np.exp(-(x/decayTiming)**decayShape)) * (initVal*decayAmp)
#                ax.plot(t,thresh,color='0.8',alpha=0.25)
#thresh = threshold - (1 - np.exp(-(x/threshDecayTiming)**threshDecayShape)) * (threshold*threshDecayAmp)
#ax.plot(t,thresh,'k',lw=2)


## fit model parameters
respRateFilePath = fileIO.getFile('Load respRate',fileType='*.npy')
respRateData = np.load(respRateFilePath)
respRateMean = np.nanmean(np.nanmean(respRateData,axis=1),axis=0)
respRateSem = np.nanstd(np.nanmean(respRateData,axis=1),axis=0)/(len(respRateData)**0.5)

fracCorrFilePath = fileIO.getFile('Load fracCorr',fileType='*.npy')
fracCorrData = np.load(fracCorrFilePath)
fracCorrMean = np.nanmean(np.nanmean(fracCorrData,axis=1),axis=0)
fracCorrSem = np.nanstd(np.nanmean(fracCorrData,axis=1),axis=0)/(len(fracCorrData)**0.5)

trialsPerCondition = 5000
targetSide = (1,0) # (-1,1,0)
maskOnset = [0,2,3,4,6,np.nan]
optoOnset = [np.nan]

alphaRange = slice(0,0.5,0.05)
nuRange = slice(2,12,1)
sigmaRange = slice(0.1,2,0.1)
decayRange = slice(0,0.5,0.05)
inhibRange = slice(0,0.5,0.05)
thresholdRange = slice(4,18,1)
trialEndRange = slice(24,25,1)

fitParamRanges = (alphaRange,nuRange,sigmaRange,decayRange,inhibRange,thresholdRange,trialEndRange)
fixedParams = (signals,targetSide,maskOnset,optoOnset,trialsPerCondition,respRateMean,fracCorrMean)

fit = fitModel(fitParamRanges,fixedParams,finish=False)

alpha,nu,sigma,decay,inhib,threshold,trialEnd = fit

trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,alpha,nu,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000,record=True)

result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)
responseRate = result['responseRate']
fractionCorrect = result['fractionCorrect']


# compare fit to data
xticks = [mo/120*1000 for mo in maskOnset[:-1]]+[67,83]
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-2]]+['target\nonly','no\nstimulus']
xlim = [-8,92]

for mean,sem,model,ylim,ylabel in  zip((respRateMean,fracCorrMean),(respRateSem,fracCorrSem),(responseRate,fractionCorrect),((0,1.02),(0.4,1)),('Response Rate','Fraction Correct')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xticks,mean,'o',mec='k',mfc='none',ms=8,mew=2,label='mice')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k')
    ax.plot(xticks,model,'o',mec='r',mfc='none',ms=8,mew=2,label='model')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()


# out of sample fits
leaveOneOutFits = []
nconditions = len(respRateMean)
for i in range(nconditions):
    print('fitting leave out condition '+str(i+1)+' of '+str(nconditions))
    if i==nconditions-1:
        ts = [s for s in targetSide if s!=0]
        mo = maskOnset
        rr = respRateMean[:-1]
        fc = fracCorrMean[:-1]
    else:
        ts = targetSide
        mo = [m for j,m in enumerate(maskOnset) if j!=i]
        rr,fc = [np.array([d for j,d in enumerate(data) if j!=i]) for data in (respRateMean,fracCorrMean)]
    fixedParams=(signals,ts,mo,optoOnset,trialsPerCondition,rr,fc)
    leaveOneOutFits.append(fitModel(fitParamRanges,fixedParams,finish=False))

outOfSampleRespRate = []
outOfSampleFracCorr = []    
for i in range(nconditions):
    sigma,decay,inhib,threshold,trialEnd = leaveOneOutFits[i]
    trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000,record=True)
    result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)
    outOfSampleRespRate.append(result['responseRate'][i])
    outOfSampleFracCorr.append(result['fractionCorrect'][i])

for diff,ylim,ylabel in  zip((outOfSampleRespRate-responseRate,outOfSampleFracCorr-fractionCorrect),([-0.2,0.2],[-0.2,0.2]),('$\Delta$ Response Rate','$\Delta$ Fraction Correct')):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,110],[0,0],'k--')
    ax.plot(xticks,diff,'ko',ms=8)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
responseRate = outOfSampleRespRate
fractionCorrect = outOfSampleFracCorr


# example model traces
trialInd = 0
for side,lbl in zip((1,0),('target right','no stim')):
    sideTrials = trialTargetSide==side
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in maskOn:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,150],[threshold,threshold],'k--')
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(mo) else trialMaskOnset==mo
        trial = np.where(sideTrials & maskTrials)[0][trialInd]
        ax.plot(t,Rrecord[trial],'r',label='R')
        ax.plot(t,Lrecord[trial],'b',label='L')
        for axside in ('right','top','left','bottom'):
            ax.spines[axside].set_visible(False)
        ax.tick_params(direction='out',right=False,top=False,left=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([])
        ax.set_xlim([0,150])
        ax.set_ylim([-1.05*threshold,1.05*threshold])
        ax.set_xlabel('Time (ms)')
        title = lbl
        if not np.isnan(mo):
            title += ' + mask (' + str(int(round(mo*dt))) + ' ms)'
        title += ', decision = '
        if response[trial]==-1:
            title += 'left'
        elif response[trial]==1:
            title += 'right'
        else:
            title += 'none'
        ax.set_title(title)
        ax.legend(loc='right')
        plt.tight_layout()


# masking reaction time
for data,ylim,ylabel in  zip((responseRate,fractionCorrect),((0,1),(0.4,1)),('Response Rate','Fraction Correct')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xticks,data,'ko')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rt = []
for side in targetSide:
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in maskOn:
        for optoOn in optoOnset:
            rt.append(dt*np.median(result[side][mo][optoOn]['responseTime']))
ax.plot(xticks,rt,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask onset relative to target onset (ms)')
ax.set_ylabel('Median decision time (ms)')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for respTime,clr,lbl in zip(('responseTimeCorrect','responseTimeIncorrect'),('k','0.5'),('correct','incorrect')):
    rt = []
    for side in targetSide:
        maskOn = [np.nan] if side==0 else maskOnset
        for mo in maskOn:
            for optoOn in optoOnset:
                if side!=0 and mo!=0:
                    rt.append(dt*np.median(result[side][mo][optoOn][respTime]))
                else:
                    rt.append(np.nan)
    ax.plot(xticks,rt,'o',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask onset relative to target onset (ms)')
ax.set_ylabel('Median decision time (ms)')
ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(2,1,2)
clrs = np.zeros((len(maskOnset)-1,3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,1,len(maskOnset)-2))[::-1,:3]
lbls = xticklabels[1:-1]
ntrials = []
rt = []
for maskOn,clr in zip(maskOnset[1:],clrs):
    trials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
    trials = trials & (trialTargetSide>0)
    ntrials.append(trials.sum())
    respTrials = trials & (response!=0)
    rt.append(responseTime[respTrials].astype(float)*dt)
    c = (trialTargetSide==response)[respTrials]
    p = []
    for i in t:
        j = (rt[-1]>=i) & (rt[-1]<i+dt)
        p.append(np.sum(c[j])/np.sum(j))
    ax.plot(t+dt/2,p,'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([0,200])
ax.set_ylim([0,1.02])
ax.set_xlabel('Model decision time (ms)')
ax.set_ylabel('Probability Correct')

ax = fig.add_subplot(2,1,1)
for r,n,clr,lbl in zip(rt,ntrials,clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([0,200])
ax.set_ylim([0,1.02])
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=8,loc='upper left')
plt.tight_layout()


# opto masking
maskOnset = [0,2,np.nan]
optoOnset = [0,2,4,6,8,10,12,14,16,np.nan]

trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=10000)

result = analyzeSession(targetSide,maskOnset,optoOnset,trialTargetSide,trialMaskOnset,trialOptoOnset,response,responseTime)

xticks = np.array(optoOnset)*dt
xticks[-1] = xticks[-2] + 2*dt
xticklabels = [int(round(x)) for x in xticks[:-1]]+['no\nopto']
for measure,ylim,ylabel in  zip(('responseRate','fractionCorrect','responseTime'),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean decision time (ms)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,side,mo,clr in zip(('target only','target + mask','mask only','no stim'),(1,1,1,0),(np.nan,2,0,np.nan),'cbkm'):
        if measure!='fractionCorrect' or 'target' in lbl:
            d = []
            for optoOn in optoOnset:
                    if measure=='responseTime':
                        d.append(dt*np.mean(result[side][mo][optoOn][measure]))
                    else:
                        d.append(result[side][mo][optoOn][measure])
            ax.plot(xticks[:-1],d[:-1],color=clr,label=lbl)
            ax.plot(xticks[-1],d[-1],'o',color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([0,xticks[-1]+dt])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    if measure=='responseRate':
        ax.legend(loc='upper left')
    plt.tight_layout()


# unilateral opto
maskOnset = [np.nan]
optoOnset = [0]




