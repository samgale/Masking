# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from maskTaskModelUtils import getInputSignals,fitModel,runSession,analyzeSession


baseDir = r"\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Sam"

## input signals
signals,t,dt = getInputSignals(psthFilePath=os.path.join(baseDir,'Analysis','popPsth.pkl'))

# signals,t,dt = getInputSignals()

fig = plt.figure(figsize=(4,9))
n = 2+len(signals['mask']['contra'].keys())
axs = []
ymin = 0
ymax = 0
i = 0
for sig in signals:
    for mo in signals[sig]['contra']:
        ax = fig.add_subplot(n,1,i+1)
        for hemi,clr in zip(('ipsi','contra'),'br'):
            p = signals[sig][hemi][mo]
            lbl = hemi if i==0 else None
            ax.plot(t,p,clr,alpha=0.5,label=lbl)
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
        ax.set_title(title)
        if i==0:
            ax.legend()
        axs.append(ax)
        i += 1
for ax in axs:
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,500])
    ax.set_ylim([1.05*ymin,1.05*ymax])
plt.tight_layout()


## fit model parameters
trialsPerCondition = 500
targetSide = (1,) # (1,0) (-1,1,0)
optoOnset = [np.nan]
optoSide = [0]

# mice
maskOnset = [0,2,3,4,6,np.nan]

respRateData = np.load(os.path.join(baseDir,'Analysis','respRate_mice.npz'))
respRateMean = respRateData['mean'][:-1]
respRateSem = respRateData['sem'][:-1]

fracCorrData = np.load(os.path.join(baseDir,'Analysis','fracCorr_mice.npz'))
fracCorrMean = fracCorrData['mean'][:-1]
fracCorrSem = fracCorrData['sem'][:-1]

reacTimeData = np.load(os.path.join(baseDir,'Analysis','reacTime_mice.npz'))
reacTimeMean = reacTimeData['mean'][:-1] / dt
reacTimeSem = reacTimeData['sem'][:-1] / dt

reacTimeCorrectData = np.load(os.path.join(baseDir,'Analysis','reacTimeCorrect_mice.npz'))
reacTimeCorrectMean = reacTimeCorrectData['mean'][:-1] / dt
reacTimeCorrectSem = reacTimeCorrectData['sem'][:-1] / dt

reacTimeIncorrectData = np.load(os.path.join(baseDir,'Analysis','reacTimeIncorrect_mice.npz'))
reacTimeIncorrectMean = reacTimeIncorrectData['mean'][:-1] / dt
reacTimeIncorrectSem = reacTimeIncorrectData['sem'][:-1] / dt

toUse = [True,True,True,True,True,True]
maskOnset = list(np.array(maskOnset)[toUse])
respRateMean = respRateMean[toUse]
respRateSem = respRateSem[toUse]
fracCorrMean = fracCorrMean[toUse]
fracCorrSem = fracCorrSem[toUse]
reacTimeMean = reacTimeMean[toUse]
reacTimeSem = reacTimeSem[toUse]
reacTimeCorrectMean = reacTimeCorrectMean[toUse]
reacTimeCorrectSem = reacTimeCorrectSem[toUse]
reacTimeIncorrectMean = reacTimeIncorrectMean[toUse]
reacTimeIncorrectSem = reacTimeIncorrectSem[toUse]

# humans
maskOnset = [0,2,4,6,8,10,12,np.nan]

respRateData = np.load(os.path.join(baseDir,'Analysis','respRate_humans.npz'))
respRateMean = respRateData['mean'][:-1]
respRateSem = respRateData['sem'][:-1]

fracCorrData = np.load(os.path.join(baseDir,'Analysis','fracCorr_humans.npz'))
fracCorrMean = fracCorrData['mean'][:-1]
fracCorrSem = fracCorrData['sem'][:-1]

reacTimeData = np.load(os.path.join(baseDir,'Analysis','reacTime_humans.npz'))
reacTimeMean = reacTimeData['mean'][:-1] / dt
reacTimeSem = reacTimeData['sem'][:-1] / dt

reacTimeCorrectData = np.load(os.path.join(baseDir,'Analysis','reacTimeCorrect_humans.npz'))
reacTimeCorrectMean = reacTimeCorrectData['mean'][:-1] / dt
reacTimeCorrectSem = reacTimeCorrectData['sem'][:-1] / dt

reacTimeIncorrectData = np.load(os.path.join(baseDir,'Analysis','reacTimeIncorrect_humans.npz'))
reacTimeIncorrectMean = reacTimeIncorrectData['mean'][:-1] / dt
reacTimeIncorrectSem = reacTimeIncorrectData['sem'][:-1] / dt

toUse = [True,True,True,True,False,False,False,True]
maskOnset = list(np.array(maskOnset)[toUse])
respRateMean = respRateMean[toUse]
respRateSem = respRateSem[toUse]
fracCorrMean = fracCorrMean[toUse]
fracCorrSem = fracCorrSem[toUse]
reacTimeMean = reacTimeMean[toUse]
reacTimeSem = reacTimeSem[toUse]
reacTimeCorrectMean = reacTimeCorrectMean[toUse]
reacTimeCorrectSem = reacTimeCorrectSem[toUse]
reacTimeIncorrectMean = reacTimeIncorrectMean[toUse]
reacTimeIncorrectSem = reacTimeIncorrectSem[toUse]
   

# simple model (no normalization)
tauIRange = slice(0,1,1)
alphaRange = slice(0,1,1)
etaRange = slice(0,1,1)
sigmaRange = slice(0.2,1.1,0.1)
tauARange = slice(0.5,5,0.5)
inhibRange = slice(0.5,1.05,0.05)
thresholdRange = slice(0.1,1.5,0.1)
trialEndRange = slice(trialEndMax,trialEndMax+1,1)

# [0, 0 ,0 , 0.4, 2.5, 1, 0.7, 24]

# with dynamic divisive normalization
tauIRange = slice(0.3,1.2,0.1)
alphaRange = slice(0.05,0.25,0.05)
etaRange = slice(1,2,1)
sigmaRange = slice(0.4,1.3,0.1)
tauARange = slice(2,9,0.5)
inhibRange = slice(0.6,1.05,0.05)
thresholdRange = slice(0.5,1.6,0.1)
trialEndRange = slice(trialEndMax,trialEndMax+1,1)

# [0.5, 0.05, 1, 1, 4.5, 0.8,  1, 24]

# fit with reaction times
tauIRange = slice(0.3,1.2,0.2)
alphaRange = slice(0.05,0.2,0.05)
etaRange = slice(1,2,1)
sigmaRange = slice(0.2,1.2,0.1)
tauARange = slice(1,10,1)
decayRange = slice(0,1.1,0.2)
inhibRange = slice(0,1,0.1)
thresholdRange = slice(0.6,2,0.1)
trialEndRange = slice(78,79,1)
postDecisionRange = slice(6,30,6)

# [0.0, 0.0, 0.0, 0.4, 1.0, 0.8, 0.0, 1.4, 214.0, 0.0] old no reac time fit
# [0.0, 0.0, 0.0, 0.5, 6.0, 0.2, 0.3, 1.8, 300.0, 18.0] old reac time fit


## fit
fitParamRanges = (tauIRange,alphaRange,etaRange,sigmaRange,tauARange,decayRange,inhibRange,thresholdRange,trialEndRange,postDecisionRange)
fixedParams = (signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,fracCorrMean,reacTimeMean)

fit = fitModel(fitParamRanges,fixedParams,finish=False)


## get best fit params from cluster output
files = glob.glob(os.path.join(baseDir,'HPC','*.npz'))
print(len(files))
modelError = None
for f in files:
    d = np.load(f)
    if modelError is None or d['error'] < modelError:
        fit = d['params']
        modelError = d['error']
    d.close()

# [0.5, 0.1, 1.0, 0.9, 6.0, 0.6, 0.6, 1.0, 60.0, 13.0] # mouse
# [0.5, 0.15,1.0, 0.9, 4.5, 0.8, 0.6, 1.0, 60.0, 15.0]
# [0.5, 0.05, 1.0, 1.2, 6.5, 0.6, 0.6, 1.2, 60.0, 13.0]
# [0.5, 0.05, 1.0, 1.3, 9.5, 1.0, 0.9, 0.8, 60.0, 13.5]
# [0.5, 0.05, 1.0, 1.1, 10.0, 1.0, 1.0, 0.7, 60.0, 14.0]
# [0.5, 0.05, 1.0, 1.3, 10.0, 1.0, 0.9, 0.8, 60.0, 13.5]
# [0.5, 0.05, 1.0, 1.1, 8.5, 1.0, 0.8, 0.8, 60.0, 14.0]
# [0.5, 0.05, 1.0, 1.3, 6.0, 0.4, 0.4, 1.5, 60.0, 12.0]
# [1.0, 0.1, 1.0, 0.9, 8.0, 0.7, 0.7, 0.8, 60.0, 14.0]
# [0.5, 0.05, 1.0, 1.2, 8.0, 0.5, 0.6, 1.1, 60.0, 12.0]
# [1.5, 0.15, 1.0, 0.3, 6.0, 0.8, 0.9, 1.0, 288.0, 20.0] # human
# [0.5, 0.15, 1.0, 0.3, 2.0, 1.0, 1.0, 1.4, 288.0, 21.5]
# [1.5, 0.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.4, 288.0, 22.5]
# [2.0, 0.05, 1.0, 0.8, 6.0, 1.0, 1.0, 1.4, 288.0, 16.5]
# [2.0, 0.1, 1.0, 0.4, 6.0, 0.7, 0.8, 1.3, 288.0, 16.0]
# [2.0, 0.15, 1.0, 0.3, 7.0, 0.7, 0.8, 0.9, 288.0, 12.0]


## run model using best fit params
tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision = fit

trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,trialsPerCondition=100000,record=True)

result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord)
responseRate = result['responseRate']
fractionCorrect = result['fractionCorrect']
responseTimeMedian= result['responseTimeMedian'] + postDecision


# compare fit to data
xticks = [mo/120*1000 for mo in maskOnset[:-1]+[maskOnset[-2]+2,maskOnset[-2]+4]]
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-2]]+['target\nonly','no\nstimulus']

xticks = [mo/120*1000 for mo in maskOnset[:-1]+[maskOnset[-2]+2]]
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-1]]+['target\nonly']

xlim = [-8,xticks[-1]+8]

for mean,sem,model,ylim,ylabel in  zip((respRateMean,fracCorrMean,reacTimeMean*dt),(respRateSem,fracCorrSem,reacTimeSem*dt),(responseRate,fractionCorrect,responseTimeMedian*dt),((0,1.02),(0.4,1),None),('Response Rate','Fraction Correct','Reaction Time (ms)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xticks,mean,'o',mec='k',mfc='none',ms=12,mew=2,label='mice')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k')
    ax.plot(xticks,model,'o',mec='r',mfc='none',ms=12,mew=2,label='model')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=14)
    if ylabel=='Fraction Correct':
        ax.set_xticks(xticks[1:])
        ax.set_xticklabels(xticklabels[1:])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(fontsize=12)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    plt.tight_layout()
    

# leave one out fits
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
    
#[array([ 0.5,  0.1,  1. ,  0.9,  4.5,  0.8,  0.9, 24. ]),
# array([ 0.6 ,  0.1 ,  1.  ,  1.  ,  4.  ,  0.75,  1.  , 24.  ]),
# array([ 0.6 ,  0.05,  1.  ,  1.1 ,  6.  ,  0.9 ,  0.9 , 24.  ]),
# array([ 0.6 ,  0.05,  1.  ,  1.1 ,  6.  ,  0.9 ,  0.9 , 24.  ]),
# array([ 0.6 ,  0.05,  1.  ,  1.1 ,  5.  ,  0.9 ,  1.  , 24.  ]),
# array([ 0.6,  0.1,  1. ,  1. ,  7.5,  0.9,  0.7, 24. ]),
# array([ 0.6 ,  0.05,  1.  ,  1.2 ,  5.  ,  1.  ,  1.1 , 24.  ])]

outOfSampleRespRate = []
outOfSampleFracCorr = []    
for i in range(nconditions):
    tauI,alpha,eta,sigma,tauA,inhib,threshold,trialEnd = leaveOneOutFits[i]
    trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,inhib,threshold,trialEnd,trialsPerCondition=100000,record=True)
    result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
    outOfSampleRespRate.append(result['responseRate'][i])
    outOfSampleFracCorr.append(result['fractionCorrect'][i])
    
for mean,sem,model,ylim,ylabel in  zip((respRateMean,fracCorrMean),(respRateSem,fracCorrSem),(outOfSampleRespRate,outOfSampleFracCorr),((0,1.02),(0.4,1)),('Response Rate','Fraction Correct')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xticks,mean,'o',mec='k',mfc='none',ms=12,mew=2,label='mice')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k')
    ax.plot(xticks,model,'o',mec='r',mfc='none',ms=12,mew=2,label='model (leave-one-out fits)')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=14)
    if ylabel=='Fraction Correct':
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    plt.tight_layout()

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


# example model traces
for side,lbl in zip((1,),('target right',)):#((1,0),('target right','no stim')):
    sideTrials = trialTargetSide==side
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in [6]:#maskOn:
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(mo) else trialMaskOnset==mo
        trials = np.where(sideTrials & maskTrials)[0]
        for trial in trials[5:7]:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot([0,trialEnd*dt],[threshold,threshold],'k--')
            ax.plot(t[t<trialEnd*dt],Lrecord[trial],'b',lw=2,label='Ipsilateral')
            ax.plot(t[t<trialEnd*dt],Rrecord[trial],'r',lw=2,label='Contralateral')
            for axside in ('right','top','left'):
                ax.spines[axside].set_visible(False)
            ax.tick_params(direction='out',right=False,top=False,left=False,labelsize=16)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,threshold])
            ax.set_yticklabels([0,'threshold'])
            ax.set_xlim([0,trialEnd*dt])
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


# masking decision time
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rt = []
for side in targetSide:
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in maskOn:
        rt.append(np.median(result[side][mo][optoOnset[0]][optoSide[0]]['responseTime']))
ax.plot(xticks,dt*(np.array(rt)+postDecision),'ko')
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
                    rt.append(np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))
                else:
                    rt.append(np.nan)
            else:
                if respTime=='responseTime':
                    rt.append(np.nan)
                else:
                    rt.append(np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))
    ax.plot(xticks,dt*(np.array(rt)+postDecision),'o',mec=mec,mfc=mfc,ms=12,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Median Reaction Time (ms)',fontsize=16)
ax.legend()
plt.tight_layout()


binWidth = 25
bins = np.arange(0,650,binWidth)
binWidth = 600
bins = np.arange(0,3000,binWidth)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,trialEnd*dt],[0.5,0.5],'k--')
clrs = np.zeros((len(maskOnset)-1,3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-2))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:-2]]+['target only']
ntrials = []
rt = []
rtCorrect = []
rtIncorrect = []
for maskOn,clr in zip(maskOnset[1:],clrs):
    trials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
    trials = trials & (trialTargetSide>0)
    ntrials.append(trials.sum())
    respTrials = trials & (response!=0)
    c = (trialTargetSide==response)[respTrials]
    rt.append(dt*(responseTime[respTrials].astype(float)+postDecision))
    rtCorrect.append(dt*(responseTime[respTrials][c].astype(float)+postDecision))
    rtIncorrect.append(dt*(responseTime[respTrials][~c].astype(float)*dt+postDecision))
    fc = []
    # for i in t[t>45]:
    #     j = (rt[-1]>=i) & (rt[-1]<i+dt)
    #     fc.append(np.sum(c[j])/np.sum(j))
    binTrials = []
    for i in bins:
        j = (rt[-1]>=i) & (rt[-1]<i+binWidth)
        fc.append(np.sum(c[j])/np.sum(j))
        binTrials.append(np.sum(j))
    i = np.array(binTrials)>14
    ax.plot(bins[i]+binWidth/2,np.array(fc)[i],'-',color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
# ax.set_xticks([0,50,100,150,200])
# ax.set_xlim([50,200])
ax.set_ylim([0.2,1])
ax.set_xlabel('Reaction Time (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,n,clr,lbl in zip(rt,ntrials,clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([0,200])
ax.set_ylim([0,1.02])
ax.set_xlabel('Decision Time (ms)',fontsize=16)
ax.set_ylabel('Cumulative Probability',fontsize=16)
ax.legend(title='mask onset',fontsize=11,loc='upper left')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(('no stim','mask only'),('0.5','g')):
    trials = trialTargetSide==0 if lbl=='no stim' else trialMaskOnset==0
    respTrials = trials & (response!=0)
    r = responseTime[respTrials].astype(float)*dt
    s = np.sort(r)
    c = [np.sum(r<=i)/len(s) for i in s]
    ax.plot(s,c,color=clr,label=lbl)
for rc,ri,clr,lbl in zip(rtCorrect,rtIncorrect,clrs,lbls):
    for r,ls in zip((rc,ri),('-','--')):
        s = np.sort(r)
        c = [np.sum(r<=i)/len(s) for i in s]
        l = lbl+', correct' if ls=='-' else lbl+', incorrect'
        ax.plot(s,c,ls,color=clr,label=l)
#ax.plot(-1,-1,'-',color='0.5',label='correct')
#ax.plot(-1,-1,'--',color='0.5',label='incorrect')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
ax.set_xticks([0,50,100,150,200])
ax.set_xlim([0,200])
ax.set_ylim([0,1.02])
ax.set_xlabel('Model Decision Time (ms)',fontsize=16)
ax.set_ylabel('Cumulative Probability',fontsize=16)
#ax.legend(loc='lower right',fontsize=11)
plt.tight_layout()


# balance of evidence
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
evidenceCorr = []
evidenceIncorr = []
for side in targetSide:
    maskOn = [np.nan] if side==0 else maskOnset
    for mo in maskOn:
        if not (side==0 or mo==0):
            evidence = []
            for evLbl in ('evidenceLeftCorrect','evidenceRightCorrect','evidenceLeftIncorrect','evidenceRightIncorrect'):
                ev = result[side][mo][optoOnset[0]][optoSide[0]][evLbl] #/ threshold
                # ev[ev>1] = 1
                # ev[ev<0] = 0
                evidence.append(ev)
            evLcorr,evRcorr,evLincorr,evRincorr = evidence
            evidenceCorr.append(np.mean(evRcorr - evLcorr))
            evidenceIncorr.append(np.mean(evLincorr - evRincorr))
ax.plot(xticks[1:],evidenceCorr,'go')
ax.plot(xticks[1:],evidenceIncorr,'mo')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask onset relative to target onset (ms)')
ax.set_ylabel('Balance of evidence')
plt.tight_layout()

trialEvidence = np.array([R-L for L,R in zip(Lrecord,Rrecord)])
evidenceResponse = []
evidenceCorrect = []
evidenceIncorrect = []
for side in (1,):
    sideTrials = trialTargetSide==1
    mo = [np.nan] if side==0 else maskOnset
    for maskOn in mo:
        maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
        trials = sideTrials & maskTrials
        responded = response[trials]!=0
        evidenceResponse.append(trialEvidence[trials][responded])
        if side!=0 and maskOn!=0:
            correct = response[trials]==side
            evidenceCorrect.append(trialEvidence[trials][responded & correct])
            evidenceIncorrect.append(trialEvidence[trials][responded & ~correct])


for er,ec,ei in zip(evidenceResponse,evidenceCorrect,evidenceIncorrect):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.nanmean(er,axis=0),'k')
    ax.plot(np.nanmean(ec,axis=0),'g')
    ax.plot(np.nanmean(ei,axis=0),'m')



# opto masking
maskOnset = [0,2,np.nan]
optoOnset = list(range(2,11))+[np.nan]
optoSide = [0]
optoLatency = 1

## run model using best fit params
trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,trialsPerCondition=100000,optoLatency=optoLatency,record=False)

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
    for lbl,side,mo,clr in zip(('target only','target + mask','mask only'),(1,1,1),(np.nan,2,0),'kbg'):
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
    ax.tick_params(direction='out',right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Simulated Inhibition Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
#    if measure=='responseRate':
#        ax.legend(loc='upper left',fontsize=12)
    plt.tight_layout()
    
# combine with masking data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xticks[2:],responseRate[[1,3,4,5]],'o',mec='r',mfc='none',mew=2,ms=12,label='masking') 
d = np.array([result[1][np.nan][optoOn][optoSide[0]]['responseRate'] for optoOn in optoOnset]) 
ax.plot(xticks,d[[0,2,4,6,8,9]],'o',mec='b',mfc='none',mew=2,ms=12,label='inhibition')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Response Rate',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
ax.plot(xticks[2:],fractionCorrect[[1,3,4,5]],'o',mec='r',mfc='none',mew=2,ms=12,label='masking')
d = np.array([result[1][np.nan][optoOn][optoSide[0]]['fractionCorrect'] for optoOn in optoOnset])
ax.plot(xticks,d[[0,2,4,6,8,9]],'o',mec='b',mfc='none',mew=2,ms=12,label='inhibition')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0.4,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
#ax.legend(fontsize=14)
plt.tight_layout()


# unilateral opto
maskOnset = [np.nan]
optoOnset = [0]
optoSide = [-1,0,1]

trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,inhib,threshold,trialEnd,trialsPerCondition=100000)

result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)


# behav
signalsCorr = createInputSignals(fileIO.getFile('Load popPsth correct',fileType='*.pkl'))[0]
signalsIncorr = createInputSignals(fileIO.getFile('Load popPsth incorrect',fileType='*.pkl'))[0]

maskOnset = [2,np.nan]
optoOnset = [np.nan]
optoSide = [0]

respRate = []
fracCorr = []
rtAll = []
rtCorr = []
rtIncorr = []
for sig in (signals,signalsCorr,signalsIncorr):
    trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = runSession(sig,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,inhib,threshold,trialEnd,trialsPerCondition=100000)
    result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
    respRate.append(result['responseRate'])
    fracCorr.append(result['fractionCorrect'])
    for i,(rt,respTime) in enumerate(zip((rtAll,rtCorr,rtIncorr),('responseTime','responseTimeCorrect','responseTimeIncorrect'))):
        rt.append([])
        for side in targetSide:
            maskOn = [np.nan] if side==0 else maskOnset
            for mo in maskOn:
                if side==0 or mo==0:
                    if respTime=='responseTime':
                        rt[-1].append(dt*np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))
                    else:
                        rt[-1].append(np.nan)
                else:
                    rt[-1].append(dt*np.median(result[side][mo][optoOnset[0]][optoSide[0]][respTime]))


for data,ylim,ylabel in  zip((respRate,fracCorr),((0,1.02),(0.4,1)),('Response Rate','Fraction Correct')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot([-1,2],[0.5,0.5],'k--')
    for d,clr,lbl in zip(data,'kgm',('all','correct','incorrect')):
        ax.plot([0,1],d[:2],'o',mec=clr,mfc='none',ms=12,mew=2,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=14)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['mask onset\n17 ms','target only'])
    ax.set_xlim([-0.3,1.3])
    ax.set_ylim(ylim)
    ax.set_ylabel('Model '+ylabel,fontsize=16)
    leg = ax.legend(title='mouse trials',fontsize=12)
    plt.setp(leg.get_title(),fontsize=12)
    plt.tight_layout()




