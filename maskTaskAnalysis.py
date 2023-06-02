# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
import fileIO
from maskTaskAnalysisUtils import MaskTaskData,fitCurve,calcLogisticDistrib,calcWeibullDistrib,inverseLogistic,inverseWeibull
   

frameRate = 120
targetSide = ('left','right')
rewardDir = (1,-1)


behavFiles = []
while True:
    files = fileIO.getFiles('choose experiments',fileType='*.hdf5')
    if len(files)>0:
        behavFiles.extend(files)
    else:
        break
    
if len(behavFiles)>0:
    exps = []
    for f in behavFiles:
        obj = MaskTaskData()
        obj.loadBehavData(f)
        exps.append(obj)


totalTrials = sum(obj.ntrials for obj in exps)
longFrames = sum(obj.longFrameTrials.sum() for obj in exps)/totalTrials
notEngaged = sum(np.sum(~obj.engaged) for obj in exps)/totalTrials
earlyMove = sum(obj.earlyMove.sum() for obj in exps)/totalTrials

sessionDur = []
for obj in exps:
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    sessionDur.append(obj.behavFrameIntervals[:obj.trialEndFrame[validTrials][-1]-1].sum()/60)
print(np.median(sessionDur),min(sessionDur),max(sessionDur))


# human reaction time task
fig = plt.figure(figsize=(6.4,4.8))
ax = fig.add_subplot(1,1,1)
targetPos = (48,480)
targetContrast = (0.4,1)
xticks = np.arange(4)
xlbl = ['eccentricity $'+str(round(pos/exps[0].pixelsPerDeg,1))+'\degree$\ncontrast '+str(c) for pos in targetPos for c in targetContrast]
clrs = plt.cm.tab20(np.linspace(0,1,len(exps)))
y = []
for obj,clr in zip(exps,clrs):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    catchTrials = obj.trialType == 'catch'
    y.append([])
    for i,pos in enumerate(targetPos):
        for j,c in enumerate(targetContrast):
            trials = validTrials & ~catchTrials & (np.absolute(obj.targetPos[:,0])==pos) & (obj.targetContrast==c) & (obj.response==1)
            rt = obj.reactionTime[trials]
            y[-1].append(np.nanmedian(rt))
    ax.plot(xticks,y[-1],color=clr,alpha=0.5)
mean = np.nanmean(y,axis=0)
sem = np.nanstd(y,axis=0)/(len(y)**0.5)
ax.plot(xticks,mean,'ko',ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xlbl)
ax.set_xlim([-0.5,3.5])
ax.set_yticks(np.arange(200,700,100))
ax.set_ylim([200,500])
ax.set_ylabel('Reaction time (ms)',fontsize=12)
plt.tight_layout()
    

fig = plt.figure(figsize=(3.6,4.8))
ax = fig.add_subplot(1,1,1)
targetPos = (48,480)
targetContrast = (0.4,1)
xticks = (0,1)
xlbl = ['eccentricity $0\degree$\ncontrast '+str(c) for c in targetContrast]
clrs = plt.cm.tab20(np.linspace(0,1,len(exps)))
y = []
for obj,clr in zip(exps,clrs):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    catchTrials = obj.trialType == 'catch'
    y.append([])
    for j,c in enumerate(targetContrast):
        trials = validTrials & ~catchTrials & (obj.targetContrast==c)
        rt = obj.reactionTime[trials]
        y[-1].append(np.nanmedian(rt))
    ax.plot(xticks,y[-1],color=clr,alpha=0.5)
mean = np.nanmean(y,axis=0)
sem = np.nanstd(y,axis=0)/(len(y)**0.5)
ax.plot(xticks,mean,'ko',ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xlbl)
ax.set_xlim([-0.5,1.5])
ax.set_yticks(np.arange(200,700,100))
ax.set_ylim([200,500])
ax.set_ylabel('Reaction time (ms)',fontsize=12)
plt.tight_layout()


# target duration
stimLabels = ('targetOnly','catch')
targetFrames = np.array([0,1,2,4,12])
ntrials = np.full((len(exps),2,len(targetFrames)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
medianReacTime = respRate.copy()
medianReacTimeCorrect = respRate.copy()
medianReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        stimTrials = validTrials & (obj.trialType==stim)
        for j,tf in enumerate(targetFrames):
            tfTrials = stimTrials  & (obj.targetFrames==tf)
            if tfTrials.sum()>0:  
                for i,rd in enumerate(rewardDir):
                    trials = tfTrials & (obj.rewardDir==rd) if stim=='targetOnly' else tfTrials
                    ntrials[n,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,i,j] = respTrials.sum()/trials.sum()
                    medianReacTime[n,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][tf] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        medianReacTimeCorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        medianReacTimeIncorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][~correctTrials]

ntable = [[],[]]
for i,(med,mn,mx) in enumerate(zip(np.median(ntrials,axis=0),np.min(ntrials,axis=0),np.max(ntrials,axis=0))):
    for j in range(ntrials.shape[-1]):
        ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal))
                    
xticks = list(targetFrames/frameRate*1000)
xticklabels = ['no\nstimulus'] + [str(int(round(x))) for x in xticks[1:]]
xlim = [-5,105]

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),(125,475)),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot(xlim,[0.5,0.5],'k--')
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.nansum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.5)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko',ms=12)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if data is fracCorr:
        ax.set_xticks(xticks[1:])
        ax.set_xticklabels(xticklabels[1:])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Duration (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    plt.tight_layout() 


# target contrast
stimLabels = ('targetOnly','catch')
targetContrast = np.unique(np.round(exps[0].targetContrast,decimals=4))
ntrials = np.full((len(exps),len(rewardDir),len(targetContrast)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
medianReacTime = respRate.copy()
medianReacTimeCorrect = respRate.copy()
medianReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        stimTrials = np.in1d(obj.trialType,('targetOnly','targetOnlyGoRight','targetOnlyGoLeft')) if stim=='targetOnly' else obj.trialType==stim
        stimTrials = stimTrials & validTrials
        for j,tc in enumerate(targetContrast):
            tcTrials = stimTrials  & (np.round(obj.targetContrast,decimals=4)==tc)
            if tcTrials.sum()>0:  
                for i,rd in enumerate(rewardDir):
                    trials = tcTrials & (obj.rewardDir==rd) if stim=='targetOnly' else tcTrials
                    ntrials[n,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,i,j] = respTrials.sum()/trials.sum()
                    medianReacTime[n,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][tc] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        medianReacTimeCorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        medianReacTimeIncorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][tc] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][tc] = obj.reactionTime[respTrials][~correctTrials]

ntable = [[],[]]
for i,(med,mn,mx) in enumerate(zip(np.median(ntrials,axis=0),np.min(ntrials,axis=0),np.max(ntrials,axis=0))):
    for j in range(ntrials.shape[-1]):
        ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal))
                    
xticks = targetContrast
xticklabels = ['no\nstimulus'] + [str(round(x,3)) for x in targetContrast[1:]]
xlim = [-0.05*targetContrast.max(),1.05*targetContrast.max()]
rtRange = (0,2000) if exps[0].rigName=='human' else (125,475)

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),rtRange),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot(xlim,[0.5,0.5],'k--')
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.5)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko',ms=12)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if not obj.useContrastStaircase:
        if data is fracCorr:
            ax.set_xticks(xticks[1:])
            ax.set_xticklabels(xticklabels[1:])
        else:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Contrast',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    plt.tight_layout()

# staircase
xunit = 'time' # 'trials' or 'time'
if obj in exps:
    if obj.useContrastStaircase:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x = np.arange(obj.targetContrast.size) + 1 if xunit=='trials' else obj.stimStart / obj.frameRate
        catch = obj.targetContrast == 0
        ax.plot(x[catch],obj.targetContrast[catch],'ko',ms=4)
        tc = obj.targetContrast.copy()
        tc[catch] = np.nan
        ax.plot(x,tc,'ko-',ms=4)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        xlbl = 'Trial' if xunit=='trials' else 'Time (s)'
        ax.set_xlabel('Trial',fontsize=16)
        ax.set_ylabel('Target Contrast',fontsize=16)
        plt.tight_layout()

# contrast response rate curve fit
fitX = np.arange(targetContrast[0],targetContrast[-1]+0.001,0.001)
contrastThreshLevel = 0.9
for i,obj in enumerate(exps):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    meanLR = np.mean(respRate[i],axis=0)
    notNan = ~np.isnan(meanLR)
    ax.plot(xticks[notNan],meanLR[notNan],'ko',ms=12)
    n = np.sum(ntrials[i],axis=0)
    n[0] = ntrials[i][0][0]
    for x,tx in zip(xticks[notNan],n[notNan]):
        ax.text(x,1.02,str(int(tx)),ha='center',va='bottom')
    for func,inv,clr,lbl in zip((calcLogisticDistrib,calcWeibullDistrib),
                                (inverseLogistic,inverseWeibull),
                                'gm',('Logistic','Weibull')):
        try:
            bounds = ((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf))
            fitParams = fitCurve(func,targetContrast[notNan],meanLR[notNan],bounds=bounds)
        except:
            fitParams = None
        if fitParams is not None:
            ax.plot(fitX,func(fitX,*fitParams),clr,label=lbl)
            a,b = fitParams[:2]
            contrastThresh = inv(b+(a-b)*contrastThreshLevel,*fitParams)
            ax.plot([contrastThresh]*2,[0,1],'--',color=clr,label=str(int(contrastThreshLevel*100))+'% max at '+str(round(contrastThresh,3)))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if not obj.useContrastStaircase:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim((0,1))
    ax.set_xlabel('Target Contrast',fontsize=16)
    ax.set_ylabel('Response Rate',fontsize=16)
    ax.legend()
    plt.tight_layout()
        
    
# masking
stimLabels = ('maskOnly','mask','targetOnly','catch')
maskOnset = np.unique(exps[0].maskOnset)
ntrials = np.full((len(exps),len(rewardDir),len(maskOnset)+2),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
probGoRight = respRate.copy()
meanVisRating = respRate.copy()
meanVisRatingResp = respRate.copy()
meanVisRatingCorrect = respRate.copy()
meanVisRatingIncorrect = respRate.copy()
visRatingMedianReacTime = respRate.copy()
visRatingMedianReacTimeCorrect = respRate.copy()
visRatingMedianReacTimeIncorrect = respRate.copy()
medianReacTime = respRate.copy()
medianReacTimeCorrect = respRate.copy()
medianReacTimeIncorrect = respRate.copy()
medianVelocity = respRate.copy()
medianVelocityCorrect = respRate.copy()
medianVelocityIncorrect = respRate.copy()
reacTime,velocity,visRatingResp = [[{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))] for _ in range(3)]
reacTimeCorrect,velocityCorrect,visRatingCorrect = [[{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))] for _ in range(3)]
reacTimeIncorrect,velocityIncorrect,visRatingIncorrect = [[{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))] for _ in range(3)]
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        stimTrials = validTrials & (obj.trialType==stim)
        for j,mo in enumerate(maskOnset):
            moTrials = stimTrials  & (obj.maskOnset==mo)
            if moTrials.sum()>0:
                if stim=='targetOnly':
                    j = -2
                elif stim=='catch':
                    j = -1  
                for i,rd in enumerate(rewardDir):
                    trials = moTrials & (obj.rewardDir==rd) if stim in ('targetOnly','mask') else moTrials
                    ntrials[n,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,i,j] = respTrials.sum()/trials.sum()
                    probGoRight[n,i,j] = np.sum(obj.responseDir[respTrials]==1)/respTrials.sum()
                    medianReacTime[n,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][mo] = obj.reactionTime[respTrials]
                    medianVelocity[n,i,j] = np.nanmedian(obj.movementVelocity[respTrials])
                    velocity[n][stim][rd][mo] = obj.movementVelocity[respTrials]
                    if hasattr(obj,'visRating'):
                        meanVisRating[n,i,j] = obj.visRatingScore[trials].mean()
                        meanVisRatingResp[n,i,j] = obj.visRatingScore[respTrials].mean()
                        visRatingMedianReacTime[n,i,j] = np.nanmedian(obj.visRatingReactionTime[respTrials])
                        visRatingResp[n][stim][rd][mo] = obj.visRatingScore[respTrials]
                    if stim in ('targetOnly','mask'):
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        medianReacTimeCorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        medianReacTimeIncorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][~correctTrials]
                        medianVelocityCorrect[n,i,j] = np.nanmedian(obj.movementVelocity[respTrials][correctTrials])
                        medianVelocityIncorrect[n,i,j] = np.nanmedian(obj.movementVelocity[respTrials][~correctTrials])
                        velocityCorrect[n][stim][rd][mo] = obj.movementVelocity[respTrials][correctTrials]
                        velocityIncorrect[n][stim][rd][mo] = obj.movementVelocity[respTrials][~correctTrials]
                        if hasattr(obj,'visRating'):
                            meanVisRatingCorrect[n,i,j] = obj.visRatingScore[respTrials][correctTrials].mean()
                            meanVisRatingIncorrect[n,i,j] = obj.visRatingScore[respTrials][~correctTrials].mean()
                            visRatingMedianReacTimeCorrect[n,i,j] = np.nanmedian(obj.visRatingReactionTime[respTrials][correctTrials])
                            visRatingMedianReacTimeIncorrect[n,i,j] = np.nanmedian(obj.visRatingReactionTime[respTrials][~correctTrials])
                            visRatingCorrect[n][stim][rd][mo] = obj.visRatingScore[respTrials][correctTrials]
                            visRatingIncorrect[n][stim][rd][mo] = obj.visRatingScore[respTrials][~correctTrials]

ntable = [[],[]]
for i,(med,mn,mx) in enumerate(zip(np.median(ntrials,axis=0),np.min(ntrials,axis=0),np.max(ntrials,axis=0))):
    for j in range(ntrials.shape[-1]):
        ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal))

xticks = np.concatenate((maskOnset,[maskOnset[-1]+2,maskOnset[-1]+4]))/frameRate*1000
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-2]]+['target\nonly','no\nstim']
xlim = [-8,(maskOnset[-1]+6)/frameRate*1000]

# single experiment
for n in range(len(exps)):
    fig = plt.figure(figsize=(6,9))
    for i,(data,ylim,ylabel) in enumerate(zip((respRate[n],fracCorr[n],medianReacTime[n],visRatingMedianReacTime[n]),((0,1.02),(0.4,1.02),None,None),('Response Rate','Fraction Correct','Median reaction time (ms)','Median rating time (ms)'))):
        ax = fig.add_subplot(4,1,i+1)
        for d,clr in zip(data,'rb'):
            ax.plot(xticks,d,'o',color=clr)
        if i==0:
            meanLR = np.mean(data,axis=0)
        else:
            meanLR = np.nansum(data*respRate[n],axis=0)/np.sum(respRate[n],axis=0)
        ax.plot(xticks,meanLR,'ko')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if i==2:
            ax.set_xlabel('Mask onset relative to target onset (ms)')
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    
# population
for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median Reaction Time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot(xlim,[0.5,0.5],'k--')
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    else:
        meanLR = np.mean(data,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.5)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko',ms=12)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if data is fracCorr:
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    plt.tight_layout()
    

# save data for model
for data,lbl in zip((respRate,fracCorr,medianReacTime,medianReacTimeCorrect,medianReacTimeIncorrect),
                    ('respRate','fracCorr','reacTime','reacTimeCorrect','reacTimeIncorrect')):
    if data is fracCorr:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    else:
        meanLR = np.nanmean(data,axis=1)
    m = np.nanmean(meanLR,axis=0)
    s = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    np.savez(fileIO.saveFile('Save '+lbl,fileType='*.npz'),mean=m,sem=s)
    

# accuracy vs mean reaction time
fc = np.sum(fracCorr*respRate,axis=1)/np.sum(respRate,axis=1)
rt = np.nanmean(medianReacTime,axis=1)

clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i,clr,lbl in zip(range(1,len(maskOnset)+1),clrs,lbls):
    x = rt[:,i]
    y = fc[:,i]
    ax.plot(x,y,'o',mec=clr,mfc=clr)
    slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
    rx = np.array([min(x),max(x)])
    ax.plot(rx,slope*rx+yint,'-',color=clr,label=lbl+', r='+str(round(rval,2))+', p='+str(round(pval,3)))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
# ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
ax.set_xlabel('Reaction Time (ms)',fontsize=14)
ax.set_ylabel('Fraction Correct',fontsize=14)
leg = ax.legend(title='mask onset',loc='lower left',fontsize=10)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = rt[:,-2]
y = fc[:,1]
ax.plot(x,y,'o',mec=clr,mfc=clr)
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
rx = np.array([min(x),max(x)])
ax.plot(rx,slope*rx+yint,'-',color='k',label='r='+str(round(rval,2))+', p='+str(round(pval,2)))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_ylim([0.4,1])
ax.set_xlabel('Reaction time, target only (ms)',fontsize=14)
ax.set_ylabel('Fraction correct, mask onset 17 ms',fontsize=14)
leg = ax.legend(loc='upper right',fontsize=10)
plt.tight_layout()
    
    
# visibility rating
for vr,lbl in zip((meanVisRatingResp,meanVisRatingCorrect,meanVisRatingIncorrect),('all trials','correct','incorrect')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xlim,[0,0],'k--')
    meanLR = np.nanmean(vr,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.5)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko',ms=12)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks([-1,0,1])
    ax.set_yticklabels(['Target not\nvisible','Unsure','Target\nvisible'])
    ax.set_xlim(xlim)
    ax.set_ylim([-1.02,1.02])
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_title(lbl)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xlim,[0,0],'k--')
for vr,mec,mfc,lbl in zip((meanVisRatingResp,meanVisRatingCorrect,meanVisRatingIncorrect),('k','k','0.5'),('none','k','0.5'),('all responses','correct','incorrect')):
    meanLR = np.nanmean(vr,axis=1)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    if vr is meanVisRatingResp:
        for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
            ax.plot(xticks,d,color=clr,alpha=0.5)
    ax.plot(xticks,mean,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'-',color=mec)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([-1,0,1])
ax.set_yticklabels(['Target\nnot visible','Unsure','Target\nvisible'])
ax.set_xlim(xlim)
ax.set_ylim([-1.02,1.02])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xlim,[0,0],'k--')
maskOnlyVr = np.nanmean(meanVisRatingResp,axis=1)[:,0]
targetOnlyCorrectVr = np.nanmean(meanVisRatingCorrect,axis=1)[:,-2]
for vr,mec,mfc,lbl in zip((meanVisRatingResp,meanVisRatingCorrect,meanVisRatingIncorrect),('k','k','0.5'),('none','k','0.5'),('all responses','correct','incorrect')):
    meanLR = np.nanmean(vr,axis=1)
    meanLR -= maskOnlyVr[:,None]
    meanLR /= (targetOnlyCorrectVr-maskOnlyVr)[:,None]
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    if vr is meanVisRatingResp:
        for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
            ax.plot(xticks,d,color=clr,alpha=0.5)
    ax.plot(xticks,mean,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'-',color=mec)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([-1,0,1])
ax.set_yticklabels(['Target\nnot visible','Unsure','Target\nvisible'])
ax.set_xlim(xlim)
ax.set_ylim([-1.02,1.02])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([-1.02,1.02],[0.5,0.5],'k--')
vr,fc = [np.nanmean(d,axis=1).flatten() for d in (meanVisRatingResp,fracCorr)]
ax.plot(vr,fc,'o',mec='k',mfc='none',mew=2,ms=12,alpha=0.5)
i = np.argsort(vr)
vr = vr[i]
fc = fc[i]
notNan = ~np.isnan(vr) & ~np.isnan(fc)
x = np.arange(-1,1.02,0.01)
slope,yint,rval,pval,stderr = scipy.stats.linregress(vr[notNan],fc[notNan])
ax.plot(x,slope*x+yint,'k',lw=2)
r,p = scipy.stats.pearsonr(vr[notNan],fc[notNan])
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks([-1,0,1])
ax.set_xticklabels(['Target\nnot visible','Unsure','Target\nvisible'])
ax.set_xlim([-1.02,1.02])
ax.set_ylim([0.4,1.02])
ax.set_xlabel('Visibility Rating',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for n in range(len(exps)):
    vr,rtc,rti = [np.nanmean(d[n],axis=0) for d in (meanVisRatingResp,medianReacTimeCorrect,medianReacTimeIncorrect)]
    if n==0:
        ax.plot(vr,rtc,'o',mec='k',mfc='none',mew=2,ms=12,label='correct')
        ax.plot(vr,rti,'o',mec='0.5',mfc='none',mew=2,ms=12,label='incorrect')
    else:
        ax.plot(vr,rtc,'o',mec='k',mfc='none',mew=2,ms=12)
        ax.plot(vr,rti,'o',mec='0.5',mfc='none',mew=2,ms=12)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks([-1,0,1])
ax.set_xticklabels(['Target not\nvisible','Unsure','Target\nvisible'])
ax.set_xlim([-1.02,1.02])
ax.set_xlabel('Visibility Rating',fontsize=16)
ax.set_ylabel('Median Reaction Time (ms)',fontsize=16)
ax.legend()
plt.tight_layout()

# visibility rating roc
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']

visRatingHitRate = np.full((len(exps),len(xticks),4),np.nan)
visRatingHitRate[:,:,0] = 0
visRatingHitRate[:,:,-1] = 1
visRatingFalseAlarmRate = visRatingHitRate.copy()
visRatingAOC = np.full((len(exps),len(xticks)),np.nan)
for i,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for stim in ('mask','targetOnly'):
        stimTrials = validTrials & (obj.trialType==stim)
        for j,mo in enumerate(maskOnset):
            trials = stimTrials  & (obj.maskOnset==mo)
            if trials.sum()>0:
                if stim=='targetOnly':
                    j = -2
                respTrials = trials & (~np.isnan(obj.responseDir))
                correctTrials = obj.response[respTrials]==1
                for k,s in zip((1,2),(0,-1)):
                    h = obj.visRatingScore[respTrials] > s
                    nc = np.sum(correctTrials)
                    ni = np.sum(~correctTrials)
                    visRatingHitRate[i,j,k] = np.sum(h[correctTrials]) / nc
                    visRatingFalseAlarmRate[i,j,k] = np.sum(h[~correctTrials]) / ni
                    if nc == 0:
                        visRatingHitRate[i,j,k] = 0
                    elif ni == 0:
                        visRatingFalseAlarmRate[i,j,k] = 0         
            visRatingAOC[i,j] = np.trapz(visRatingHitRate[i,j],visRatingFalseAlarmRate[i,j])


fig = plt.figure()
for i,(hr,far) in enumerate(zip(visRatingHitRate,visRatingFalseAlarmRate)):
    ax = fig.add_subplot(4,4,i+1)
    ax.plot([0,1],[0,1],'--',color='0.5')
    for h,fa,clr in zip(hr[1:-1],far[1:-1],clrs):
        ax.plot(fa,h,color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.set_aspect('equal')
    if i==0:
        ax.set_ylabel('Hit Rate')
    else:
        ax.set_yticklabels([])
    if i==15:
        ax.set_xlabel('False Alarm Rate')
    else:
        ax.set_xticklabels([])
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xlim,[0.5,0.5],'k--')
for d,clr in zip(visRatingAOC,plt.cm.tab20(np.linspace(0,1,visRatingAOC.shape[0]))):
    ax.plot(xticks,d,color=clr,alpha=0.5)
mean = np.nanmean(visRatingAOC,axis=0)
sem = np.nanstd(visRatingAOC,axis=0)/(visRatingAOC.shape[0]**0.5)
ax.plot(xticks,mean,'ko',ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_ylim([0.4,1.02])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Predicted Fraction Correct',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1.5],[0,1.5],'--',color='0.5')
vrMean = np.nanmean(visRatingAOC,axis=0)[1:-1]
vrSem = (np.nanstd(visRatingAOC,axis=0)/(visRatingAOC.shape[0]**0.5))[1:-1]
fcMean = np.nanmean(fracCorr,axis=(0,1))[1:-1]
fcSem = (np.nanstd(fracCorr,axis=(0,1))/(fracCorr.shape[0]**0.5))[1:-1]
for vr,fc,clr,lbl in zip(vrMean,fcMean,clrs,lbls):
    ax.plot(vr,fc,'o',mec=clr,mfc='none',mew=2,ms=12,label=lbl)
for x,y,s,clr in zip(vrMean,fcMean,vrSem,clrs):
    ax.plot([x-s,x+s],[y,y],color=clr)
for x,y,s,clr in zip(vrMean,fcMean,fcSem,clrs):
    ax.plot([x,x],[y-s,y+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(0,1.1,0.2))
ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_xlim([0.4,1.02])
ax.set_ylim([0.4,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Predicted Fraction Correct',fontsize=16)
ax.set_ylabel('Observed Fraction Correct',fontsize=16)
ax.legend(loc='lower right',title='mask onset')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1.5],[0,1.5],'--',color='0.5',label='unity')
vr = visRatingAOC.flatten()
fc = np.nanmean(fracCorr,axis=1).flatten()
ax.plot(vr,fc,'o',mec='k',mfc='none',mew=2,ms=12,alpha=0.5)
i = np.argsort(vr)
vr = vr[i]
fc = fc[i]
notNan = ~np.isnan(vr) & ~np.isnan(fc)
x = np.arange(-0.4,1.02,0.01)
slope,yint,rval,pval,stderr = scipy.stats.linregress(vr[notNan],fc[notNan])
r,p = scipy.stats.pearsonr(vr[notNan],fc[notNan])
ax.plot(x,slope*x+yint,'k',lw=2,label='linear fit (r='+str(round(r,2))+', p='+'{0:1.1e}'.format(p)+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(0,1.1,0.2))
ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_xlim([0.4,1.02])
ax.set_ylim([0.4,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Predicted Fraction Correct',fontsize=16)
ax.set_ylabel('Observed Fraction Correct',fontsize=16)
ax.legend(loc='lower right',fontsize=10)
plt.tight_layout()

    
# pooled trials across mice
fc = np.nansum(fracCorr*respRate,axis=1)/np.nansum(respRate,axis=1)
nresp = np.sum(ntrials*respRate,axis=1)

maskingPooledRespRate = np.sum(respRate*ntrials/ntrials.sum(axis=(0,1)),axis=(0,1))
maskingPooledRespRateCi = [[c/n for c in scipy.stats.binom.interval(0.95,n,p)] for n,p in zip(ntrials.sum(axis=(0,1)),maskingPooledRespRate)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xlim,[0.5,0.5],'k--')
maskingPooledFracCorr = np.nansum(fc*nresp/nresp.sum(axis=0),axis=0)
ax.plot(xticks,maskingPooledFracCorr,'o',color='k',ms=12)
maskingPooledFracCorrCi = []
for x,n,p in zip(xticks,nresp.sum(axis=0),maskingPooledFracCorr):
    if p>0:
        s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
        ax.plot([x,x],s,color='k')
        maskingPooledFracCorrCi.append(s)
    else:
        maskingPooledFracCorrCi.append(np.nan)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks[1:-1])
ax.set_xticklabels(xticklabels[1:-1])
ax.set_xlim(xlim)
ax.set_ylim([0.4,1])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct (pooled trials)',fontsize=16)
plt.tight_layout()

# wt vgat comparison
vgatInd = slice(0,8)
wtInd = slice(8,16)
for data,ylim,ylabel in zip((respRate,fracCorr),((0,1),(0.4,1)),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot(xlim,[0.5,0.5],'k--')
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for i,mfc,lbl in zip((vgatInd,wtInd),('k','none'),('VGAT-ChR2','Wild-type')):
        mean = np.nanmean(meanLR[i],axis=0)
        sem = np.nanstd(meanLR[i],axis=0)/(meanLR[i].shape[0]**0.5)
        ax.plot(xticks,mean,'o',mec='k',mfc=mfc,ms=12,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if data is fracCorr:
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    if ylabel=='Response Rate':
        ax.legend()
    plt.tight_layout()
    
alpha = 0.05
for data,lbl in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):        
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    pvals = []
    for x,y in zip(meanLR[vgatInd].T,meanLR[wtInd].T):
        pvals.append(scipy.stats.ranksums(x,y)[1])
    pvalsCorr = multipletests(pvals,alpha=alpha,method='fdr_bh')[1]
    print(lbl,pvalsCorr)
    
# stats
alpha = 0.05
for data,title in zip((respRate,fracCorr,meanVisRatingResp),('Response Rate','Fraction Correct','Visibility Rating')):        
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    p = scipy.stats.kruskal(*meanLR.T,nan_policy='omit')[1]
    pmat = np.full((meanLR.shape[1],)*2,np.nan)
    for i,x in enumerate(meanLR.T):
        for j,y in enumerate(meanLR.T):
            if j>i and not np.all(np.isnan(x+y)):
                pmat[i,j] = scipy.stats.ranksums(x,y)[1]
                
    pvals = pmat.flatten()
    notnan = ~np.isnan(pvals)
    pvalsCorr = np.full(pvals.size,np.nan)
    pvalsCorr[notnan] = multipletests(pvals[notnan],alpha=alpha,method='fdr_bh')[1]
    pmatCorr = np.reshape(pvalsCorr,pmat.shape)
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.nanmin(pvalsCorr))),alpha)
    clim = np.log10(lim)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color=np.array((255, 251, 204))/255)
    im = ax.imshow(np.log10(pmatCorr),cmap=cmap,clim=clim)
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            if pmatCorr[i,j] < alpha:
                ax.plot(j,i,'*',color='r',ms=8)
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(np.arange(len(xticklabels)))
    ax.set_yticklabels(xticklabels)
    ax.set_xlim([-0.5,len(xticklabels)-0.5])
    ax.set_ylim([-0.5,len(xticklabels)-0.5])
    ax.set_xlabel('Mask onset (ms)',fontsize=14)
    ax.set_ylabel('Mask onset (ms)',fontsize=14)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.ax.tick_params(labelsize=10) 
    legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
    cb.set_ticks(legticks)
    cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
    ax.set_title(title+' Comparisons (p value)',fontsize=14)
    plt.tight_layout()
    
# comparison to chance
meanLR = np.nansum(fracCorr*respRate,axis=1)/np.sum(respRate,axis=1)
n = ntrials.sum(axis=1)
pval = np.full(n.shape,np.nan)
for i in range(len(pval)):
    for j in range(1,6):
        pval[i,j] = scipy.stats.binom.cdf(0.5*n[i,j],n[i,j],meanLR[i,j])
    
# spearman correlation of accuracy vs mask onset
rs = []
ps = []
for fc in np.sum(fracCorr*respRate,axis=1)/np.sum(respRate,axis=1):
    r,p = scipy.stats.spearmanr(np.arange(5),fc[1:6])
    rs.append(r)
    ps.append(p)
    
# performance by target side
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']
for data,ylabel in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1],'--',color='0.5')
    for i,clr,lbl in zip(range(1,len(maskOnset)+1),clrs,lbls):
        ax.plot(data[:,0,i],data[:,1,i],'o',mec=clr,mfc=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,1.02])
    ax.set_ylim([0,1.02])
    ax.set_aspect('equal')
    ax.set_xlabel('Target Left '+ylabel,fontsize=14)
    ax.set_ylabel('Target Right '+ylabel,fontsize=14)
    if ylabel=='Response Rate':
        leg = ax.legend(title='mask onset',loc='upper left',fontsize=10)
        plt.setp(leg.get_title(),fontsize=10)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'--',color='0.5')
meanLR = np.nansum(probGoRight*respRate,axis=1)/np.sum(respRate,axis=1)
for i,clr,lbl in zip(range(1,len(maskOnset)+1),clrs,lbls):
    x = meanLR[:,0]
    y = meanLR[:,i]
    ax.plot(x,y,'o',mec=clr,mfc=clr)
    slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
    rx = np.array([min(x),max(x)])
    ax.plot(rx,slope*rx+yint,'-',color=clr,label='r='+str(round(rval,2))+', p='+'{:.0e}'.format(pval))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Prob. Go Right (Mask Only)',fontsize=14)
ax.set_ylabel('Prob. Go Right (Target Either Side)',fontsize=14)
leg = ax.legend(loc='upper left',fontsize=10)
#plt.setp(leg.get_title(),fontsize=10)
plt.tight_layout()
    
# reaction time on correct and incorrect trials
for measures,ylbl in zip(((medianReacTimeCorrect,medianReacTimeIncorrect,medianReacTime),(medianVelocityCorrect,medianVelocityIncorrect,medianVelocity)),('Median Reaction Time (ms)','Median Movement Speed (mm/s)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for data,clr,lbl in zip(measures,('k','0.5','k'),('correct','incorrect','other')):
        meanLR = np.nanmean(data,axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if lbl=='other':
            xt = [xticks[0],xticks[-1]]
            ax.plot(xt,mean[[0,-1]],'o',mec=clr,mfc='none',ms=12,label=lbl)
            for x,m,s in zip(xt,mean[[0,-1]],sem[[0,-1]]):
                ax.plot([x,x],[m-s,m+s],'-',color=clr)
        else:
            ax.plot(xticks,mean,'o',color=clr,ms=12,label=lbl)
            for x,m,s in zip(xticks,mean,sem):
                ax.plot([x,x],[m-s,m+s],'-',color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylbl,fontsize=16)
    legLoc = 'upper right' if exps[0].rigName=='human' else 'upper left'
    ax.legend(loc=legLoc,fontsize=12)
    plt.tight_layout()
    
norm = False
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
targetOnlyCorrectRt = np.nanmean(medianReacTimeCorrect,axis=1)[:,-2]
for rt,mec,mfc,lbl in zip((medianReacTime,medianReacTimeCorrect,medianReacTimeIncorrect),('k','k','0.5'),('none','k','0.5'),('all responses','correct','incorrect')):
    meanLR = np.nanmean(rt,axis=1)
    if norm:
        meanLR /= targetOnlyCorrectRt[:,None]
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    if rt is medianReacTime:
        for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
            ax.plot(xticks,d,color=clr,alpha=0.5)
    ax.plot(xticks,mean,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'-',color=mec)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
if exps[0].rigName=='human':
    ax.set_ylim([400,1900])
else:
    ax.set_yticks(np.arange(100,600,100))
    ax.set_ylim([100,500])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Reaction Time (ms)',fontsize=16)
legLoc = 'upper right' if exps[0].rigName=='human' else 'upper left'
ax.legend(loc=legLoc,fontsize=12)
plt.tight_layout()
    

clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']

rtLim = [400,2200] if exps[0].rigName=='human' else [150,410]
for measures,alim,albl in zip(((medianReacTimeCorrect,medianReacTimeIncorrect),(medianVelocityCorrect,medianVelocityIncorrect)),(rtLim,[0,65]),('Reaction Time (ms)','Movement Speed (mm/s)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    rc,ri = [np.nanmean(d,axis=1) for d in measures]
    for j,(clr,lbl) in enumerate(zip(clrs,lbls)):
        ax.plot(rc[:,j+1],ri[:,j+1],'o',color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    if exps[0].rigName=='human':
        ax.set_xticks(np.arange(0,2500,500))
        ax.set_yticks(np.arange(0,2500,500))
    if 'Speed' in albl:
        ax.set_yticks(np.arange(0,100,20))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Correct Trials',fontsize=16)
    ax.set_ylabel('Incorrect Trials',fontsize=16)
    ax.set_title(albl,fontsize=16)
    if 'Time' in albl:
        leg = ax.legend(title='mask onset',loc='lower right',fontsize=11)
        plt.setp(leg.get_title(),fontsize=11)
    plt.tight_layout()

# fraction correct vs reaction time
if exps[0].rigName == 'human':
    binWidth = 600
    bins = np.arange(0,3000,binWidth)
else:
    binWidth = 50
    bins = np.arange(0,650,binWidth)
quantileSize = 0.1
quantiles = np.arange(quantileSize,1.01,quantileSize)
quantiles = [0.1,0.3,0.7,0.9,1]
rt = []
rtCorrect = []
rtIncorrect = []
vel = []
velCorrect = []
velIncorrect = []
vrResp = []
vrCorrect = []
vrIncorrect = []
fcBinned = []
binTrials = []
quantileBins = []
fcQuantiled = []
quantileTrials = []
for mo in list(maskOnset[1:])+[maskOnset[0]]:
    stim = 'mask' if mo>0 else 'targetOnly'
    rt.append([])
    rtCorrect.append([])
    rtIncorrect.append([])
    vel.append([])
    velCorrect.append([])
    velIncorrect.append([])
    vrResp.append([])
    vrCorrect.append([])
    vrIncorrect.append([])
    correct = np.zeros(bins.size-1)
    incorrect = correct.copy()
    for i in range(len(exps)):
        for rd in rewardDir:
            for r,d in zip((rt,rtCorrect,rtIncorrect,vel,velCorrect,velIncorrect,vrResp,vrCorrect,vrIncorrect),
                           (reacTime,reacTimeCorrect,reacTimeIncorrect,velocity,velocityCorrect,velocityIncorrect,visRatingResp,visRatingCorrect,visRatingIncorrect)):
                if exps[0].rigName != 'human' and d in (visRatingResp,visRatingCorrect,visRatingIncorrect):
                    continue
                t = d[i][stim][rd][mo]
                r[-1].extend(t[~np.isnan(t)])
            c,ic = [np.histogram(r[i][stim][rd][mo],bins)[0] for r in (reacTimeCorrect,reacTimeIncorrect)]
            correct += c
            incorrect += ic
    fcBinned.append(correct/(correct+incorrect))
    binTrials.append(correct+incorrect)  
for i,t in enumerate(rt):
    quantileBins.append(np.concatenate(([np.min(t)],np.quantile(t,quantiles))))
    c,ic = [np.histogram(r,quantileBins[-1])[0] for r in (rtCorrect[i],rtIncorrect[i])]
    fcQuantiled.append(c/(c+ic))
    quantileTrials.append(c+ic)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,2500],[0.5,0.5],'k--')
for p,n,clr,lbl in zip(fcBinned,binTrials,clrs,lbls):
    i = n>14
    ax.plot(bins[:-1][i]+binWidth/2,p[i],color=clr,label=lbl)
    s = [c/n[i] for c in scipy.stats.binom.interval(0.95,n[i],p[i])]
    s[0][p[i]==1] = 1
    ax.fill_between(bins[:-1][i]+binWidth/2,s[1],s[0],color=clr,alpha=0.2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
if exps[0].rigName=='human':
    ax.set_xlim([200,2100])
else:
    ax.set_xlim([100,475])
ax.legend(title='mask onset',loc='lower left')
ax.set_ylim([0.2,1])
ax.set_xlabel('Reaction Time (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,2500],[0.5,0.5],'k--')
for b,p,n,clr,lbl in zip(quantileBins,fcQuantiled,quantileTrials,clrs,lbls):
    x = (b[:-1]+b[1:])/2
    ax.plot(x,p,color=clr,label=lbl)
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    s[0][p==1] = 1
    ax.fill_between(x,s[1],s[0],color=clr,alpha=0.2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
if exps[0].rigName=='human':
    ax.set_xlim([300,2100])
    ax.legend(loc='lower left')
else:
    ax.set_xlim([100,475])
ax.set_ylim([0.2,1])
ax.set_xlabel('Reaction Time (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

if exps[0].rigName == 'human':   
    for r,vr in zip((rt,rtCorrect,rtIncorrect),(vrResp,vrCorrect,vrIncorrect)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for t,v,clr,lbl in zip(r,vr,clrs,lbls):   
            b = np.digitize(t,bins)
            m = np.zeros(len(bins))
            s = m.copy()
            for j in range(bins.size):
                d = np.array(v)[b==j+1]
                m[j] = np.mean(d)
                s[j] = np.std(d)/(len(d)**0.5)
            ax.plot(bins[:-1]+binWidth/2,m[:-1],color=clr,label=lbl)
            ax.fill_between(bins[:-1]+binWidth/2,(m+s)[:-1],(m-s)[:-1],color=clr,alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for stim,lbl,clr,ls in zip(('catch','maskOnly'),('no stimulus','mask only'),('k','0.5'),('--','-')):
    n = ntrials[:,0,0].sum() if stim=='maskOnly' else ntrials[:,0,-1].sum()
    r = np.concatenate([reacTime[i][stim][1][0] for i in range(len(exps))])
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,ls,color=clr,label=lbl)
for r,n,clr,lbl in zip(rt,ntrials.sum(axis=(0,1))[1:1+len(maskOnset)],clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=14)
if exps[0].rigName=='human':
    ax.set_xlim([0,2500])
    leg = ax.legend(loc='lower right',fontsize=11)
else:
    ax.set_xlim([100,625])
    leg = ax.legend(loc='upper left',fontsize=11)
ax.set_ylim([0,1.02])
ax.set_xlabel('Reaction Time (ms)',fontsize=16)
ax.set_ylabel('Cumulative Probability',fontsize=16)
plt.tight_layout()

for corr,incorr,xlim,xlbl in zip((rtCorrect,velCorrect),(rtIncorrect,velIncorrect),([100,625],[0,100]),('Reaction Time (ms)','Movement Speed (mm/s)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for stim,lbl,clr in zip(('catch','maskOnly'),('no stimulus','mask only'),('0.5','g')):
        r = np.concatenate([reacTime[i][stim][1][0] for i in range(len(exps))])
        s = np.sort(r)
        c = [np.sum(r<=i)/len(s) for i in s]
        ax.plot(s,c,color=clr,label=lbl)
    for rc,ri,clr,lbl in zip(corr,incorr,clrs,lbls):
        for r,ls in zip((rc,ri),('-','--')):
            s = np.sort(r)
            c = [np.sum(r<=i)/len(s) for i in s]
            l = lbl+', correct' if ls=='-' else lbl+', incorrect'
            ax.plot(s,c,ls,color=clr,label=l)
#    ax.plot(-1,-1,'-',color='0.5',label='correct')
#    ax.plot(-1,-1,'--',color='0.5',label='incorrect')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=14)
    if exps[0].rigName=='human':
        ax.set_xlim([0,2500])
    else:
        ax.set_xlim(xlim)
    ax.set_ylim([0,1])
    ax.set_xlabel(xlbl,fontsize=16)
    ax.set_ylabel('Cumulative Probability',fontsize=16)
    if 'Time' in xlbl:
        ax.legend(loc='lower right',fontsize=9)
    plt.tight_layout()


# opto timing
stimLabels = ('targetOnly','catch')
optoOnset = np.array([4,6,8,10,12,np.nan])
ntrials = np.full((len(exps),len(stimLabels),len(rewardDir),len(optoOnset)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
medianReacTime = respRate.copy()
medianReacTimeCorrect = respRate.copy()
medianReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for s,stim in enumerate(stimLabels):
        stimTrials = validTrials & np.in1d(obj.trialType,(stim,stim+'Opto'))
        for j,optoOn in enumerate(optoOnset):
            optoTrials = stimTrials & np.isnan(obj.optoOnset) if np.isnan(optoOn) else stimTrials & (obj.optoOnset==optoOn)
            if optoTrials.sum()>0:
                for i,rd in enumerate(rewardDir):
                    trials = optoTrials & (obj.rewardDir==rd) if stim=='targetOnly' else optoTrials
                    ntrials[n,s,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,s,i,j] = respTrials.sum()/trials.sum()
                    medianReacTime[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][optoOn] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,s,i,j] = correctTrials.sum()/respTrials.sum()
                        medianReacTimeCorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        medianReacTimeIncorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][~correctTrials]
 
ntable = [[],[],[]]
for i,(med,mn,mx) in enumerate(zip(np.concatenate(np.median(ntrials,axis=0)),np.concatenate(np.min(ntrials,axis=0)),np.concatenate(np.max(ntrials,axis=0)))):
    if i<len(ntable):
        for j in range(ntrials.shape[-1]):
            ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2,3))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal))    

respAboveChancePval = np.full((len(exps),len(optoOnset)),np.nan)            
for i in range(len(exps)):
    chanceRespRate = respRate[i,-1,0,-1]
    for j in range(len(optoOnset)):
        n = ntrials[i,0,:,j].sum()
        respAboveChancePval[i,j] = scipy.stats.binom.sf(n*respRate[i,0,:,j].mean(),n,chanceRespRate)


xticks = list((optoOnset[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']
                    
for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot([8,108],[0.5,0.5],'k--')
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','no stim'),'km')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.nansum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
            if stim=='targetOnly':
                meanLR[respAboveChancePval>=0.05] = np.nan
                nAboveChance = np.sum(~np.isnan(meanLR),axis=0)
                print(nAboveChance)
                meanLR[:,nAboveChance<2] = np.nan
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        lbl = stimLbl if data is respRate else None
        for d in meanLR:
            ax.plot(xticks,d,color=clr,alpha=0.25)
        ax.plot(xticks,mean,'o',color=clr,ms=12,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Optogenetic Light Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    if data is respRate:
        ax.legend(loc='upper left',fontsize=14)
    plt.tight_layout()

# stats
alpha = 0.05
for data,title in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):       
    if data is respRate:
        meanLR = np.mean(data,axis=2)
    else:
        meanLR = np.nansum(data*respRate,axis=2)/np.sum(respRate,axis=2)
        meanLR[:,0][respAboveChancePval>=0.05] = np.nan
        meanLR[:,np.sum(~np.isnan(meanLR),axis=0)<2] = np.nan
    meanLR = np.reshape(meanLR,(meanLR.shape[0],meanLR.shape[1]*meanLR.shape[2]))
    p = scipy.stats.kruskal(*meanLR.T,nan_policy='omit')[1]
    pmat = np.full((meanLR.shape[1],)*2,np.nan)
    for i,x in enumerate(meanLR.T):
        for j,y in enumerate(meanLR.T):
            if j>i and np.nansum(x)>0 and np.nansum(y)>0:
                pmat[i,j] = scipy.stats.ranksums(x,y)[1]
                
    pvals = pmat.flatten()
    notnan = ~np.isnan(pvals)
    pvalsCorr = np.full(pvals.size,np.nan)
    pvalsCorr[notnan] = multipletests(pvals[notnan],alpha=alpha,method='fdr_bh')[1]
    pmatCorr = np.reshape(pvalsCorr,pmat.shape)
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.nanmin(pvalsCorr))),alpha)
    clim = np.log10(lim)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color=np.array((255, 251, 204))/255)
    im = ax.imshow(np.log10(pmatCorr),cmap=cmap,clim=clim)
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            if pmatCorr[i,j] < alpha:
                ax.plot(j,i,'*',color='r',ms=8)
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(2*len(xticklabels)))
    ax.set_xticklabels(2*xticklabels)
    ax.set_yticks(np.arange(2*len(xticklabels)))
    ax.set_yticklabels(2*xticklabels)
    ax.set_xlim([-0.5,2*len(xticklabels)-0.5])
    ax.set_ylim([-0.5,2*len(xticklabels)-0.5])
    ax.set_xlabel('Optogenetic light onset (ms)',fontsize=14)
    ax.set_ylabel('Optogenetic light onset (ms)',fontsize=14)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.ax.tick_params(labelsize=10)
    legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
    cb.set_ticks(legticks)
    cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
    ax.set_title(title+' Comparisons (p value)',fontsize=14)
    plt.tight_layout()

# pooled trials fraction correct
fc,rt = [np.nansum(d[:,0]*respRate[:,0],axis=1)/np.nansum(respRate[:,0],axis=1) for d in (fracCorr,medianReacTime)]
nresp = np.sum(ntrials[:,0]*respRate[:,0],axis=1)

rr = np.mean(respRate[:,0]-respRate[:,1,0,-1][:,None,None],axis=1)
bw = 0.2
bins = np.arange(-0.1,1.01,bw)
for data,ylim,ylabel in zip((fc,rt),((-0.02,1.02),None),('Fraction Correct','Reaction Time')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if ylabel=='Fraction Correct':
        ax.plot([bins[0],1],[0.5,0.5],'k--')
    ax.plot(rr,data,'o',mec='0.6',mfc='none',ms=10)
    ind = np.digitize(rr,bins)
    for i,b in enumerate(bins[:-1]):
        bi = ind==i+1
        x = b+0.5*bw
        ntotal = nresp[bi].sum()
        m = np.nansum(data[bi]*nresp[bi]/ntotal)
        if ylabel=='Fraction Correct':
            s = [c/ntotal for c in scipy.stats.binom.interval(0.95,ntotal,m)]
            print(s,ntotal)
        else:
            s = np.nanstd(data[bi])/(bi.sum()**0.5)
            s = [m-s,m+s]
        ax.plot(x,m,'ko',ms=12)
        ax.plot([x,x],s,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([-0.1,0.9])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Response Rate Relative to Chance',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
ntotal = nresp.sum(axis=0)
m = np.nansum(fc*nresp/ntotal,axis=0)
ax.plot(xticks,m,'o',color='k',ms=12)
for x,n,p in zip(xticks,ntotal,m):
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    ax.plot([x,x],s,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([8,108])
ax.set_ylim([0.4,1])
ax.set_xlabel('Optogenetic Light Onset Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct (pooled trials)',fontsize=16)
plt.tight_layout()

# pooled data combined with masking data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xticks[2:],maskingPooledRespRate[[1,3,4,5]],'o',mec='r',mfc='none',mew=2,ms=12,label='masking')
for x,s in zip(xticks[2:],np.array(maskingPooledRespRateCi)[[1,3,4,5]]):
    ax.plot([x,x],s,color='r')
m = np.sum(respRate[:,0]*ntrials[:,0]/ntrials[:,0].sum(axis=(0,1)),axis=(0,1))
ax.plot(xticks,m,'o',mec='b',mfc='none',mew=2,ms=12,label='inhibition')
for x,n,p in zip(xticks,ntrials[:,0].sum(axis=(0,1)),m):
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    ax.plot([x,x],s,color='b')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Response Rate (pooled trials)',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
ax.plot(xticks[2:],maskingPooledFracCorr[[1,3,4,5]],'o',mec='k',mfc='none',mew=2,ms=12,label='masking')
for x,s in zip(xticks[2:],np.array(maskingPooledFracCorrCi)[[1,3,4,5]]):
    ax.plot([x,x],s,color='k')
m = np.nansum(fc*nresp/nresp.sum(axis=0),axis=0)
ax.plot(xticks,m,'o',mec='k',mfc='k',mew=2,ms=12,label='inhibition')
for x,n,p in zip(xticks,nresp.sum(axis=0),m):
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    ax.plot([x,x],s,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0.4,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct (pooled trials)',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()


# opto masking
stimLabels = ('targetOnly','mask','maskOnly','catch')
optoOnset = [4,6,8,10,12,np.nan]
ntrials = np.full((len(exps),len(stimLabels),len(rewardDir),len(optoOnset)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
medianReacTime = respRate.copy()
medianReacTimeCorrect = respRate.copy()
medianReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for s,stim in enumerate(stimLabels):
        mo = 2 if stim=='mask' else 0
        stimTrials = validTrials & np.in1d(obj.trialType,(stim,stim+'Opto')) & (obj.maskOnset==mo)
        for j,optoOn in enumerate(optoOnset):
            optoTrials = stimTrials & np.isnan(obj.optoOnset) if np.isnan(optoOn) else stimTrials & (obj.optoOnset==optoOn)
            for i,rd in enumerate(rewardDir):
                trials = optoTrials & (obj.rewardDir==rd) if stim in ('targetOnly','mask') else optoTrials
                ntrials[n,s,i,j] = trials.sum()
                respTrials = trials & (~np.isnan(obj.responseDir))
                respRate[n,s,i,j] = respTrials.sum()/trials.sum()
                medianReacTime[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                reacTime[n][stim][rd][optoOn] = obj.reactionTime[respTrials]
                if stim in ('targetOnly','mask'):
                    correctTrials = obj.response[respTrials]==1
                    fracCorr[n,s,i,j] = correctTrials.sum()/respTrials.sum()
                    medianReacTimeCorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                    medianReacTimeIncorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                    reacTimeCorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][correctTrials]
                    reacTimeIncorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][~correctTrials]

ntable = [[],[],[],[],[],[]]
for i,(med,mn,mx) in enumerate(zip(np.concatenate(np.median(ntrials,axis=0)),np.concatenate(np.min(ntrials,axis=0)),np.concatenate(np.max(ntrials,axis=0)))):
    if i<len(ntable):
        for j in range(ntrials.shape[-1]):
            ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2,3))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal)) 

respAboveChancePval = np.full((len(exps),len(stimLabels)-2,len(optoOnset)),np.nan)
p = respAboveChancePval.copy()             
for i in range(len(exps)):
    chanceRespRate = respRate[i,-1,0,-1]
    for s in range(len(stimLabels)-2):
        for j in range(len(optoOnset)):
            n = ntrials[i,s,:,j].sum()
            respAboveChancePval[i,s,j] = scipy.stats.binom.sf(n*respRate[i,s,:,j].mean(),n,chanceRespRate)


xticks = list((np.array(optoOnset)[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']

for data,ylim,ylabel in zip((respRate,fracCorr),((0,1),(0.4,1)),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is fracCorr:
        ax.plot([8,108],[0.5,0.5],'k--')
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','target + mask','mask only','no stim'),'kbgm')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.nansum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
            if stim in ('targetOnly','mask'):
                meanLR[respAboveChancePval[:,i]>=0.05] = np.nan
                nAboveChance = np.sum(~np.isnan(meanLR),axis=0)
                print(nAboveChance)
                meanLR[:,nAboveChance<3] = np.nan
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        # for d in meanLR:
        #     ax.plot(xticks,d,color=clr,alpha=0.25)
        ax.plot(xticks,mean,'-o',color=clr,ms=12,label=stimLbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Optogenetic Light Onset Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    if data is respRate:
        ax.legend(loc='upper left',fontsize=14)
#    if data is fracCorr:
#        ax.legend(loc='upper center',fontsize=11)
    plt.tight_layout()
    
# stats
alpha = 0.05
for data,title in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):       
    if data is respRate:
        meanLR = np.mean(data,axis=2)
    else:
        meanLR = np.nansum(data*respRate,axis=2)/np.sum(respRate,axis=2)
        meanLR[:,:2][respAboveChancePval>=0.05] = np.nan
        meanLR[:,np.sum(~np.isnan(meanLR),axis=0)<3] = np.nan
    meanLR = np.reshape(meanLR,(meanLR.shape[0],meanLR.shape[1]*meanLR.shape[2]))
    p = scipy.stats.kruskal(*meanLR.T,nan_policy='omit')[1]
    pmat = np.full((meanLR.shape[1],)*2,np.nan)
    for i,x in enumerate(meanLR.T):
        for j,y in enumerate(meanLR.T):
            if j>i and np.nansum(x)>0 and np.nansum(y)>0:
                pmat[i,j] = scipy.stats.ranksums(x,y)[1]
                
    pvals = pmat.flatten()
    notnan = ~np.isnan(pvals)
    pvalsCorr = np.full(pvals.size,np.nan)
    pvalsCorr[notnan] = multipletests(pvals[notnan],alpha=alpha,method='fdr_bh')[1]
    pmatCorr = np.reshape(pvalsCorr,pmat.shape)
    
    fig = plt.figure(facecolor='w',figsize=(8,7))
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.nanmin(pvalsCorr))),alpha)
    clim = np.log10(lim)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color=np.array((255, 251, 204))/255)
    im = ax.imshow(np.log10(pmatCorr),cmap=cmap,clim=clim)
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            if pmatCorr[i,j] < alpha:
                ax.plot(j,i,'*',color='r',ms=8)
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(4*len(xticklabels)))
    ax.set_xticklabels(4*xticklabels)
    ax.set_yticks(np.arange(4*len(xticklabels)))
    ax.set_yticklabels(4*xticklabels)
    ax.set_xlim([-0.5,4*len(xticklabels)-0.5])
    ax.set_ylim([-0.5,4*len(xticklabels)-0.5])
    ax.set_xlabel('Optogenetic light onset (ms)',fontsize=14)
    ax.set_ylabel('Optogenetic light onset (ms)',fontsize=14)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.ax.tick_params(labelsize=10)
    legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
    cb.set_ticks(legticks)
    cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
    ax.set_title(title+' Comparisons (p value)',fontsize=14)
    plt.tight_layout()
    
# pooled trials fraction correct
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels[:2],('target only','target + mask'),'kb')):
    fc = np.nansum(fracCorr[:,i]*respRate[:,i],axis=1)/np.nansum(respRate[:,i],axis=1)
    nresp = np.sum(ntrials[:,i]*respRate[:,i],axis=1)
    ntotal = nresp.sum(axis=0)
    m = np.nansum(fc*nresp/ntotal,axis=0)
    ax.plot(xticks,m,'o',color=clr,ms=12,label=lbl)
    for x,n,p in zip(xticks,ntotal,m):
        s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
        ax.plot([x,x],s,color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([8,108])
ax.set_ylim([0.4,1])
ax.set_xlabel('Optogenetic light onset relative to target onset (ms)',fontsize=12)
ax.set_ylabel('Fraction Correct (pooled trials)',fontsize=12)
plt.tight_layout()

# pooled data combined with masking data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xticks[2:],maskingPooledRespRate[[1,3,4,5]],'o',mec='r',mfc='none',mew=2,ms=12,label='masking')
for x,s in zip(xticks[2:],np.array(maskingPooledRespRateCi)[[1,3,4,5]]):
    ax.plot([x,x],s,color='r')
m = np.sum(respRate[:,0]*ntrials[:,0]/ntrials[:,0].sum(axis=(0,1)),axis=(0,1))
ax.plot(xticks,m,'o',mec='b',mfc='none',mew=2,ms=12,label='inhibition')
for x,n,p in zip(xticks,ntrials[:,0].sum(axis=(0,1)),m):
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    ax.plot([x,x],s,color='b')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Response Rate (pooled trials)',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()

fc = np.nansum(fracCorr[:,0]*respRate[:,0],axis=1)/np.nansum(respRate[:,0],axis=1)
nresp = np.sum(ntrials[:,0]*respRate[:,0],axis=1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
ax.plot(xticks[2:],maskingPooledFracCorr[[1,3,4,5]],'o',mec='r',mfc='none',mew=2,ms=12,label='masking')
for x,s in zip(xticks[2:],np.array(maskingPooledFracCorrCi)[[1,3,4,5]]):
    ax.plot([x,x],s,color='r')
m = np.nansum(fc*nresp/nresp.sum(axis=0),axis=0)
ax.plot(xticks,m,'o',mec='b',mfc='none',mew=2,ms=12,label='inhibition')
for x,n,p in zip(xticks,nresp.sum(axis=0),m):
    s = [c/n for c in scipy.stats.binom.interval(0.95,n,p)]
    ax.plot([x,x],s,color='b')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(['-17','0','17','33','50','no mask or\ninhibition'])
ax.set_xlim([8,108])
ax.set_ylim([0.4,1])
ax.set_xlabel('Mask or Inhibition Onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct (pooled trials)',fontsize=16)
#ax.legend(fontsize=14)
plt.tight_layout()
    
# fraction correct vs reaction time
binWidth = 50
bins = np.arange(0,650,binWidth)
for ind,stim in enumerate(('mask','targetOnly')):
    rt = []
    pc = []
    for optoOn in optoOnset:
        rt.append([])
        correct = np.zeros(bins.size-1)
        incorrect = correct.copy()
        for i in range(len(exps)):
            for rd in rewardDir:
                rt[-1].extend(reacTime[i][stim][rd][optoOn])
                c,ic = [np.histogram(rt[i][stim][rd][optoOn],bins)[0] for rt in (reacTimeCorrect,reacTimeIncorrect)]
                correct += c
                incorrect += ic
        pc.append(correct/(correct+incorrect))

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    clrs = np.zeros((len(optoOnset),3))
    clrs[:-1] = plt.cm.plasma(np.linspace(0,1,len(optoOnset)-1))[::-1,:3]
    
    clrs = np.zeros((len(optoOnset),3))
    cint = 1/(len(optoOnset)-1)
    clrs[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
    clrs[:-1,2] = 1
    
    lbls = [lbl+' ms' for lbl in xticklabels[:-1]]+[xticklabels[-1]]
    for r,n,clr,lbl in zip(rt,ntrials[:,ind].sum(axis=(0,1)),clrs,lbls):
        s = np.sort(r)
        c = [np.sum(r<=i)/n for i in s]
        ax.plot(s,c,'-',color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=10)
    ax.set_xlim([100,500])
    ax.set_ylim([0,1.02])
    ax.set_ylabel('Cumulative Probability',fontsize=12)
    ax.legend(title='opto onset',loc='upper left',fontsize=8)
    
    ax = fig.add_subplot(2,1,2)
    ax.plot([0,650],[0.5,0.5],'--',color='0.8')
    for p,clr in zip(pc,clrs):
        ax.plot(bins[:-2]+binWidth/2,p[:-1],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',right=False,labelsize=10)
    ax.set_xlim([100,500])
    ax.set_ylim([0,1.02])
    ax.set_xlabel('Reaction Time (ms)',fontsize=12)
    ax.set_ylabel('Probability Correct',fontsize=12)
    plt.tight_layout()


# unilateral opto
nexps = len(exps)
stimLabels = ('target left','target right','no stim')
optoSide = ('left','right','both','no opto')
ntrials = [{stim: [] for stim in stimLabels} for _ in range(nexps)]
ntrials = np.full((nexps,len(stimLabels),len(optoSide)),np.nan)
respRate = [{stim: {respDir: [] for respDir in rewardDir} for stim in stimLabels} for _ in range(nexps)]
medianReacTime = copy.deepcopy(respRate)
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for i,stim in enumerate(stimLabels):
        if stim=='no stim':
            stimTrials = validTrials & np.in1d(obj.trialType,('catch','catchOpto')) & np.isnan(obj.rewardDir)
        else:
            rd = 1 if stim=='target left' else -1
            stimTrials = validTrials & np.in1d(obj.trialType,('targetOnly','targetOnlyOpto')) & (obj.rewardDir==rd)
        for j,opSide in enumerate(optoSide):
            if opSide=='left':
                optoTrials = stimTrials & obj.optoChan[:,0] & ~obj.optoChan[:,1]
            elif opSide=='right':
                optoTrials= stimTrials & ~obj.optoChan[:,0] & obj.optoChan[:,1]
            elif opSide=='both':
                optoTrials = stimTrials & obj.optoChan[:,0] & obj.optoChan[:,1]
            else:
                optoTrials = stimTrials & np.isnan(obj.optoOnset)
            ntrials[n][i][j] = optoTrials.sum()
            for respDir in rewardDir:
                respTrials = optoTrials & (obj.responseDir==respDir)
                respRate[n][stim][respDir].append(respTrials.sum()/optoTrials.sum())
                medianReacTime[n][stim][respDir].append(np.nanmedian(obj.reactionTime[respTrials]))

ntable = [[],[],[]]
for i,(med,mn,mx) in enumerate(zip(np.median(ntrials,axis=0),np.min(ntrials,axis=0),np.max(ntrials,axis=0))):
    for j in range(ntrials.shape[-1]):
        ntable[i].append(str(int(round(med[j])))+' ('+str(int(mn[j]))+'-'+str(int(mx[j]))+')')
ntable = np.array(ntable)
ntotal = ntrials.sum(axis=(1,2))
print(np.median(ntotal),np.min(ntotal),np.max(ntotal)) 
                   
fig = plt.figure(figsize=(6,9))
xticks = np.arange(len(optoSide))
for i,stim in enumerate(stimLabels):    
    ax = fig.add_subplot(len(stimLabels),1,i+1)
    for respDir,clr,lbl in zip(rewardDir,'rb',('move right','move left')):
        d = [respRate[n][stim][respDir] for n in range(nexps)]
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(xticks,mean,'o-',color=clr,ms=12,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xlim([-0.25,len(optoSide)-0.75])
    if i==2:
        ax.set_xticklabels(optoSide)
        ax.set_xlabel('Optogenetic Stimulus Side',fontsize=16)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,1,0.2))
    ax.set_ylim([0,0.75])
    if i==1:
        ax.set_ylabel('Fraction of Trials',fontsize=16)
    if i==0:
        ax.legend(fontsize=14)
    ax.set_title(stim,fontsize=16)
plt.tight_layout()




