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
    iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd = paramsToFit
    signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,responseRate,fractionCorrect = fixedParams
    trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition)
    result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
    respRateError = np.nansum((responseRate-result['responseRate'])**2)
    fracCorrError = np.nansum((2*(fractionCorrect-result['fractionCorrect']))**2)
    return respRateError + fracCorrError


def analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime):
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
                result[side][maskOn][optoOn] = {}
                for opSide in optoSide:
                    optoTrials = np.isnan(trialOptoOnset) if np.isnan(optoOn) else (trialOptoOnset==optoOn) & (trialOptoSide==opSide)
                    trials = sideTrials & maskTrials & optoTrials
                    responded = response[trials]!=0
                    responseRate.append(np.sum(responded)/np.sum(trials))
                    result[side][maskOn][optoOn][opSide] = {}
                    result[side][maskOn][optoOn][opSide]['responseRate'] = responseRate[-1]
                    result[side][maskOn][optoOn][opSide]['responseTime'] = responseTime[trials][responded]
                    if side!=0 and maskOn!=0:
                        correct = response[trials]==side
                        fractionCorrect.append(np.sum(correct[responded])/np.sum(responded))
                        result[side][maskOn][optoOn][opSide]['fractionCorrect'] = fractionCorrect[-1]
                        result[side][maskOn][optoOn][opSide]['responseTimeCorrect'] = responseTime[trials][responded & correct]
                        result[side][maskOn][optoOn][opSide]['responseTimeIncorrect'] = responseTime[trials][responded & (~correct)]
                    else:
                        fractionCorrect.append(np.nan)
    result['responseRate'] = np.array(responseRate)
    result['fractionCorrect'] = np.array(fractionCorrect)
    return result


def runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition,optoLatency=0,record=False):
    trialTargetSide = []
    trialMaskOnset = []
    trialOptoOnset = []
    trialOptoSide = []
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
                for opSide in optoSide:
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
                        i = int(optoOn+optoLatency)
                        if opSide <= 0:
                            Lsignal[i:] = 0
                        if opSide >= 0:
                            Rsignal[i:] = 0
                    if iDecay==0 and alpha > 0:
                        for s in (Lsignal,Rsignal):
                            i = s > 0
                            s[i] = s[i]**eta / (alpha**eta + s[i]**eta)
                            s *= alpha**eta + 1
                    for _ in range(trialsPerCondition):
                        trialTargetSide.append(side)
                        trialMaskOnset.append(maskOn)
                        trialOptoOnset.append(optoOn)
                        trialOptoSide.append(opSide)
                        result = runTrial(iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record)
                        response.append(result[0])
                        responseTime.append(result[1])
                        if record:
                            Lrecord.append(result[2])
                            Rrecord.append(result[3])
    return np.array(trialTargetSide),np.array(trialMaskOnset),np.array(trialOptoOnset),np.array(trialOptoSide),np.array(response),np.array(responseTime),Lrecord,Rrecord


# divisive normalization of input signal
@njit
def runTrial(iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(Lsignal.size,np.nan)
        Rrecord = Lrecord.copy()
    else:
        Lrecord = Rrecord = None
    L = R = 0
    iL = iR = 0
    t = 0
    response = 0
    while t<trialEnd and response==0:
        if record:
            Lrecord[t] = L
            Rrecord[t] = R
        if L > threshold and R > threshold:
            response = -1 if L > R else 1
        elif L > threshold:
            response = -1
        elif R > threshold:
            response = 1
        if iDecay > 0 and alpha > 0:
            Lsig = (Lsignal[t]**eta) / (alpha**eta + iL**eta) if Lsignal[t]>0 and iL>=0 else Lsignal[t]
            Rsig = (Rsignal[t]**eta) / (alpha**eta + iR**eta) if Rsignal[t]>0 and iR>=0 else Rsignal[t]
        else:
            Lsig = Lsignal[t]
            Rsig = Rsignal[t]
        Lnow = L
        L += (random.gauss(0,sigma) + Lsig - L - inhib*R) / decay
        R += (random.gauss(0,sigma) + Rsig - R - inhib*Lnow) / decay
        iL += (Lsignal[t] - iL) / iDecay
        iR += (Rsignal[t] - iR) / iDecay
        t += 1
    responseTime = t-1
    return response,responseTime,Lrecord,Rrecord


# simple model
@njit
def runTrial(iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(Lsignal.size,np.nan)
        Rrecord = Lrecord.copy()
    else:
        Lrecord = Rrecord = None
    L = R = 0
    t = 0
    response = 0
    while t<trialEnd and response==0:
        if record:
            Lrecord[t] = L
            Rrecord[t] = R
        if L > threshold and R > threshold:
            response = -1 if L > R else 1
        elif L > threshold:
            response = -1
        elif R > threshold:
            response = 1
        Lnow = L
        L += random.gauss(0,sigma) + Lsignal[t] - decay*L - inhib*R
        R += random.gauss(0,sigma) + Rsignal[t] - decay*R - inhib*Lnow
        t += 1
    responseTime = t-1
    return response,responseTime,Lrecord,Rrecord


# divisive normalization of integrators (Keung et al. 2020)
#@njit
#def runTrial(iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,Lsignal,Rsignal,record=False):
#    if record:
#        Lrecord = np.full(Lsignal.size,np.nan)
#        Rrecord = Lrecord.copy()
#    else:
#        Lrecord = Rrecord = None
#    L = R = 0
#    G = 0
#    t = 0
#    response = 0
#    while t<trialEnd and response==0:
#        if record:
#            Lrecord[t] = L
#            Rrecord[t] = R
#        if L > threshold and R > threshold:
#            response = -1 if L > R else 1
#        elif L > threshold:
#            response = -1
#        elif R > threshold:
#            response = 1
#        L += (random.gauss(0,sigma) + Lsignal[t]/(alpha+G) - L) / decay
#        R += (random.gauss(0,sigma) + Rsignal[t]/(alpha+G) - R) / decay
#        G += (inhib*(Lsignal[t] + Rsignal[t]) - G) / iDecay
#        t += 1
#    responseTime = t-1
#    return response,responseTime,Lrecord,Rrecord



# create model input signals from population ephys responses
popPsthFilePath = fileIO.getFile(fileType='*.pkl')
popPsth = pickle.load(open(popPsthFilePath,'rb'))

dt = 1/120*1000
trialEndTimeMax = 200
trialEndMax = int(round(trialEndTimeMax/dt))

t = np.arange(0,trialEndMax*dt+dt,dt)
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
            p = np.interp(t,popPsth['t']*1000,p)
            p -= p[t<30].mean()
            p[0] = 0
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
            
#            if alpha>0:
#                if iDecay==0:
#                    i = s > 0
#                    s[i] = s[i]**eta / (alpha**eta + s[i]**eta)
#                    s[i] *= alpha**eta + 1
#                else:
#                    sraw = s.copy()
#                    I = 0
#                    for i in range(s.size):
#                        if i > 0:
#                            I += (sraw[i-1] - I) / iDecay
#                        if s[i]>0 and I>=0:
#                            s[i] = (s[i]**eta) / (alpha**eta + I**eta)


fig = plt.figure(figsize=(4,9))
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

trialsPerCondition = 100
targetSide = (1,0) # (-1,1,0)
maskOnset = [0,2,3,4,6,np.nan]
optoOnset = [np.nan]
optoSide = [0]

iDecayRange = slice(0,1,1)
alphaRange = slice(0,1,1)
etaRange = slice(0,1,1)
sigmaRange = slice(0,0.6,0.05)
decayRange = slice(0,0.45,0.05)
inhibRange = slice(0,0.45,0.05)
thresholdRange = slice(1,11,1)
trialEndRange = slice(trialEndMax,trialEndMax+1,1)

iDecayRange = slice(0.2,1.4,0.2)
alphaRange = slice(0.05,0.35,0.05)
etaRange = slice(1,2,1)
sigmaRange = slice(0.1,1.3,0.1)
decayRange = slice(0.5,8,0.5)
inhibRange = slice(0.1,1.1,0.1) # slice(0.5,1.05,0.05)
thresholdRange = slice(0.5,2,0.1)
trialEndRange = slice(trialEndMax,trialEndMax+1,1)

fitParamRanges = (iDecayRange,alphaRange,etaRange,sigmaRange,decayRange,inhibRange,thresholdRange,trialEndRange)
fixedParams = (signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,fracCorrMean)

fit = fitModel(fitParamRanges,fixedParams,finish=False)


iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd = fit

trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000,record=True)

result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
responseRate = result['responseRate']
fractionCorrect = result['fractionCorrect']


# compare fit to data
xticks = [mo/120*1000 for mo in maskOnset[:-1]]+[67,83]
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-2]]+['target\nonly','no\nstimulus']
xlim = [-8,92]

for mean,sem,model,ylim,ylabel in  zip((respRateMean,fracCorrMean),(respRateSem,fracCorrSem),(responseRate,fractionCorrect),((0,1.02),(0.4,1)),('Response Rate','Fraction Correct')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xticks,mean,'o',mec='k',mfc='none',ms=12,mew=2,label='mice')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k')
    ax.plot(xticks,model,'o',mec='r',mfc='none',ms=12,mew=2,label='model')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=12)
    if ylabel=='Fraction Correct':
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)
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
    fixedParams=(signals,ts,mo,optoOnset,optoSide,trialsPerCondition,rr,fc)
    leaveOneOutFits.append(fitModel(fitParamRanges,fixedParams,finish=False))

outOfSampleRespRate = []
outOfSampleFracCorr = []    
for i in range(nconditions):
    iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd = leaveOneOutFits[i]
    trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000,record=True)
    result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
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
for side,lbl in zip((1,),('target right',)):#((1,0),('target right','no stim')):
    sideTrials = trialTargetSide==side
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in [np.nan]:#maskOn:
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(mo) else trialMaskOnset==mo
        trials = np.where(sideTrials & maskTrials)[0]
        for trial in trials[5:7]:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot([0,trialEndTimeMax],[threshold,threshold],'k--')
            ax.plot(t,Lrecord[trial],'b',lw=2,label='Ipsilateral')
            ax.plot(t,Rrecord[trial],'r',lw=2,label='Contralateral')
            for axside in ('right','top','left'):
                ax.spines[axside].set_visible(False)
            ax.tick_params(direction='out',right=False,top=False,left=False,labelsize=16)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,threshold])
            ax.set_yticklabels([0,'threshold'])
            ax.set_xlim([0,trialEndTimeMax])
            ax.set_ylim([-1.05*threshold,1.05*threshold])
            ax.set_xlabel('Time (ms)',fontsize=18)
            ax.set_ylabel('Decision Variable',fontsize=18)
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
#            ax.set_title(title)
            ax.legend(loc='lower right',fontsize=16)
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
        rt.append(dt*np.median(result[side][mo][optoOnset[0]][optoSide[0]]['responseTime']))
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
for respTime,mec,mfc,lbl in zip(('responseTimeCorrect','responseTimeIncorrect','responseTime',),('k','0.5','k'),('k','0.5','none'),('correct','incorrect','other')):
    rt = []
    for side in targetSide:
        maskOn = [np.nan] if side==0 else maskOnset
        for mo in maskOn:
            if side==0 or mo==0:
                if respTime=='responseTime':
                    rt.append(dt*np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))
                else:
                    rt.append(np.nan)
            else:
                if respTime=='responseTime':
                    rt.append(np.nan)
                else:
                    rt.append(dt*np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))
    ax.plot(xticks,rt,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=14)
ax.set_ylabel('Median Decision Time (ms)',fontsize=14)
#ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,200],[0.5,0.5],'k--')
clrs = np.zeros((len(maskOnset)-1,3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-2))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:-2]]+['target only']
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
    for i in t[t>45]:
        j = (rt[-1]>=i) & (rt[-1]<i+dt)
        p.append(np.sum(c[j])/np.sum(j))
    ax.plot(t[t>45]+dt/2,p,'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=12)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([50,200])
ax.set_ylim([0.2,1])
ax.set_xlabel('Decision Time (ms)',fontsize=14)
ax.set_ylabel('Fraction Correct',fontsize=14)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,n,clr,lbl in zip(rt,ntrials,clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=10)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([0,200])
ax.set_ylim([0,1.02])
ax.set_xlabel('Decision Time (ms)',fontsize=12)
ax.set_ylabel('Cumulative Probability',fontsize=12)
ax.legend(title='mask onset',fontsize=8,loc='upper left')
plt.tight_layout()


# opto masking
maskOnset = [0,2,np.nan]
optoOnset = list(range(2,11))+[np.nan]
optoSide = [0]
optoLatency = 1

trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000,optoLatency=optoLatency)

result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)

xticks = [x*dt for x in optoOnset[::2]]+[100]
xticklabels = [int(round(x)) for x in xticks[:-1]]+['no\nopto']
x = np.array(optoOnset)*dt
x[-1] = 100
for measure,ylim,ylabel in  zip(('responseRate','fractionCorrect','responseTime'),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean decision time (ms)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if measure=='fractionCorrect':
        j = 2
        ax.plot([0,xticks[-1]+dt],[0.5,0.5],'k--')
    else:
        j = 0
    for lbl,side,mo,clr in zip(('target only','target + mask','mask only','no stim'),(1,1,1,0),(np.nan,2,0,np.nan),'kbgm'):
        if measure!='fractionCorrect' or 'target' in lbl:
            d = []
            for optoOn in optoOnset:
                if measure=='responseTime':
                    d.append(dt*np.mean(result[side][mo][optoOn][optoSide[0]][measure]))
                else:
                    d.append(result[side][mo][optoOn][optoSide[0]][measure])
            ax.plot(x[:-1],d[:-1],color=clr,label=lbl)
            ax.plot(xticks[j:],np.array(d)[np.in1d(x,xticks)][j:],'o',color=clr,ms=12)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=13)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Simulated Inhibition Relative to Target Onset (ms)',fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)
#    if measure=='responseRate':
#        ax.legend(loc='upper left',fontsize=12)
    plt.tight_layout()


# unilateral opto
maskOnset = [np.nan]
optoOnset = [0]
optoSide = [-1,0,1]

trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,iDecay,alpha,eta,sigma,decay,inhib,threshold,trialEnd,trialsPerCondition=100000)

result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)






