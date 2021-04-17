# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
import fileIO
from maskTaskAnalysisUtils import MaskTaskData
   


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

    
frameRate = 120
rewardDir = (1,-1)


# target duration
stimLabels = ('targetOnly','catch')
targetFrames = np.array([0,1,2,4,12])
ntrials = np.full((len(exps),2,len(targetFrames)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
meanReacTime = respRate.copy()
meanReacTimeCorrect = respRate.copy()
meanReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
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
                    meanReacTime[n,i,j] = np.nanmean(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][tf] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        meanReacTimeCorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][correctTrials])
                        meanReacTimeIncorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][~correctTrials]
                    else:
                        break
                    
xticks = list(targetFrames/frameRate*1000)
xticklabels = ['no\nstimulus'] + [str(int(round(x))) for x in xticks[1:]]
xlim = [-5,105]

for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    meanLR = np.nanmean(data,axis=1)
    for d in meanLR:
        ax.plot(xticks[0],d[0],'o',mec='0.5',mfc='none')
        ax.plot(xticks[1:],d[1:],color='0.5')
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks[0],mean[0],'ko')
    ax.plot(xticks[1:],mean[1:],'ko-')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Duration (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout() 


# target contrast
stimLabels = ('targetOnly','catch')
targetContrast = np.array([0,0.2,0.4,0.6,1])
ntrials = np.full((len(exps),2,len(targetContrast)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
meanReacTime = respRate.copy()
meanReacTimeCorrect = respRate.copy()
meanReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        stimTrials = validTrials & (obj.trialType==stim)
        for j,tc in enumerate(targetContrast):
            tcTrials = stimTrials  & (obj.targetContrast==tc)
            if tcTrials.sum()>0:  
                for i,rd in enumerate(rewardDir):
                    trials = tcTrials & (obj.rewardDir==rd) if stim=='targetOnly' else tcTrials
                    ntrials[n,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,i,j] = respTrials.sum()/trials.sum()
                    meanReacTime[n,i,j] = np.nanmean(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][tf] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        meanReacTimeCorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][correctTrials])
                        meanReacTimeIncorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][tf] = obj.reactionTime[respTrials][~correctTrials]
                    else:
                        break
                    
xticks = targetContrast
xticklabels = ['no\nstimulus'] + [str(x) for x in targetContrast[1:]]
xlim = [-0.05,1.05]

for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    meanLR = np.nanmean(data,axis=1)
    for d in meanLR:
        ax.plot(xticks[0],d[0],'o',mec='0.5',mfc='none')
        ax.plot(xticks[1:],d[1:],color='0.5')
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks[0],mean[0],'ko')
    ax.plot(xticks[1:],mean[1:],'ko-')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Contrast (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()                  
    

# masking
stimLabels = ('mask','targetOnly','maskOnly','catch')
maskOnset = np.array([2,3,4,6,0])
ntrials = np.full((len(exps),2,len(maskOnset)+2),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
meanReacTime = respRate.copy()
meanReacTimeCorrect = respRate.copy()
meanReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        stimTrials = validTrials & (obj.trialType==stim)
        for j,mo in enumerate(maskOnset):
            moTrials = stimTrials  & (obj.maskOnset==mo)
            if moTrials.sum()>0:
                if stim=='targetOnly':
                    j = -3
                elif stim=='maskOnly':
                    j = -2
                elif stim=='catch':
                    j = -1  
                for i,rd in enumerate(rewardDir):
                    trials = moTrials & (obj.rewardDir==rd) if stim in ('targetOnly','mask') else moTrials
                    ntrials[n,i,j] = trials.sum()
                    respTrials = trials & (~np.isnan(obj.responseDir))
                    respRate[n,i,j] = respTrials.sum()/trials.sum()
                    meanReacTime[n,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][mo] = obj.reactionTime[respTrials]
                    if stim in ('targetOnly','mask'):
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        meanReacTimeCorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        meanReacTimeIncorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][~correctTrials]
                    else:
                        break
                    
#np.save(fileIO.saveFile(fileType='*.npy'),respRate)
#np.save(fileIO.saveFile(fileType='*.npy'),fracCorr)

xticks = list(maskOnset[:-1]/frameRate*1000)+[67,83,100]
xticklabels = [str(int(round(x))) for x in xticks[:-3]]+['target\nonly','mask\nonly','no\nstimulus']
xlim = [8,108]

# single experiment
for n in range(len(exps)):
    fig = plt.figure(figsize=(6,9))
    for i,(data,ylim,ylabel) in enumerate(zip((respRate[n],fracCorr[n],meanReacTime[n]),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Reaction Time (ms)'))):
        ax = fig.add_subplot(3,1,i+1)
        for d,clr in zip(data,'rb'):
            ax.plot(xticks,d,'o',color=clr)
        ax.plot(xticks,np.nanmean(data,axis=0),'ko')
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
for data,ylim,ylabel in zip((respRate,fracCorr),((0,1),(0.4,1)),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    meanLR = np.nanmean(data,axis=1)
    for d in meanLR:
        ax.plot(xticks,d,color='0.5')
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
# reaction time on correct and incorrect trials
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for data,clr,lbl in zip((meanReacTime,meanReacTimeCorrect,meanReacTimeIncorrect),'kgr',('all','correct','incorrect')):
    meanLR = np.nanmean(data,axis=1)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'o',color=clr,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask onset relative to target onset (ms)')
ax.set_ylabel('Mean reaction time (ms)')
ax.legend()
plt.tight_layout()

# fraction correct vs reaction time
binWidth = 50
bins = np.arange(0,650,binWidth)
rt = []
pc = []
for mo in maskOnset:
    stim = 'mask' if mo>0 else 'targetOnly'
    rt.append([])
    correct = np.zeros(bins.size-1)
    incorrect = correct.copy()
    for i in range(len(exps)):
        for rd in rewardDir:
            rt[-1].extend(reacTime[i][stim][rd][mo])
            c,ic = [np.histogram(rt[i][stim][rd][mo],bins)[0] for rt in (reacTimeCorrect,reacTimeIncorrect)]
            correct += c
            incorrect += ic
    pc.append(correct/(correct+incorrect))

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,1,len(maskOnset)-1))[::-1,:3]
lbls = xticklabels[:-3]+['target only']
for r,n,clr,lbl in zip(rt,ntrials.sum(axis=(0,1)),clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xlim([100,650])
ax.set_ylim([0,1.02])
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=8,loc='upper left')

ax = fig.add_subplot(2,1,2)
for p,clr in zip(pc,clrs):
    ax.plot(bins[:-1]+binWidth/2,p,color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xlim([100,650])
ax.set_ylim([0,1.02])
ax.set_xlabel('Reaction Time (ms)')
ax.set_ylabel('Probability Correct')
plt.tight_layout()


# opto timing
stimLabels = ('targetOnly','catch')
optoOnset = np.array([4,6,8,10,12,np.nan])
ntrials = np.full((len(exps),len(stimLabels),2,len(optoOnset)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
meanReacTime = respRate.copy()
meanReacTimeCorrect = respRate.copy()
meanReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
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
                    meanReacTime[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][mo] = obj.reactionTime[respTrials]
                    if stim=='targetOnly':
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,s,i,j] = correctTrials.sum()/respTrials.sum()
                        meanReacTimeCorrect[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials][correctTrials])
                        meanReacTimeIncorrect[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][~correctTrials]
                    else:
                        break

xticks = list((optoOnset[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']
                    
for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','no stim'),'cm')):
        meanLR = np.nanmean(data[:,i],axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if data is fracCorr:
            if stim=='targetOnly':
                firstValid = 2
            else:
                firstValid = 0
        else:
            firstValid = 0
        lbl = stimLbl if data is respRate else None
        ax.plot(xticks[firstValid:-1],mean[firstValid:-1],'o',color=clr)
        ax.plot(xticks[-1],mean[-1],'o',color=clr,label=lbl)
        for x,m,s in zip(xticks[firstValid:],mean[firstValid:],sem[firstValid:]):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    if data is respRate:
        ax.legend()
    plt.tight_layout()


# opto masking
stimLabels = ('mask','targetOnly','maskOnly','catch')
optoOnset = np.array([4,6,8,10,12,np.nan])
ntrials = np.full((len(exps),len(stimLabels),2,len(optoOnset)),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
meanReacTime = respRate.copy()
meanReacTimeCorrect = respRate.copy()
meanReacTimeIncorrect = respRate.copy()
reacTime = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeCorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
reacTimeIncorrect = [{stim: {rd: {} for rd in rewardDir} for stim in stimLabels} for _ in range(len(exps))]
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
    for s,stim in enumerate(stimLabels):
        mo = 2 if stim in ('mask','maskOpto') else 0
        stimTrials = validTrials & np.in1d(obj.trialType,(stim,stim+'Opto')) & (obj.maskOnset==mo)
        for j,optoOn in enumerate(optoOnset):
            optoTrials = stimTrials & np.isnan(obj.optoOnset) if np.isnan(optoOn) else stimTrials & (obj.optoOnset==optoOn)
            for i,rd in enumerate(rewardDir):
                trials = optoTrials & (obj.rewardDir==rd) if stim in ('targetOnly','targetOnlyOpto','mask','maskOpto') else optoTrials
                ntrials[n,s,i,j] = trials.sum()
                respTrials = trials & (~np.isnan(obj.responseDir))
                respRate[n,s,i,j] = respTrials.sum()/trials.sum()
                meanReacTime[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials])
                reacTime[n][stim][rd][optoOn] = obj.reactionTime[respTrials]
                if stim in ('targetOnly','targetOnlyOpto','mask','maskOpto'):
                    correctTrials = obj.response[respTrials]==1
                    fracCorr[n,s,i,j] = correctTrials.sum()/respTrials.sum()
                    meanReacTimeCorrect[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials][correctTrials])
                    meanReacTimeIncorrect[n,s,i,j] = np.nanmean(obj.reactionTime[respTrials][~correctTrials])
                    reacTimeCorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][correctTrials]
                    reacTimeIncorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][~correctTrials]
                else:
                    break

xticks = list((optoOnset[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']

for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target + mask','target only','mask only','no stim'),'bckm')):
#        meanLR = (data[:,i,0]*ntrials[:,i,0]+data[:,i,1]*ntrials[:,i,1])/ntrials[:,i].sum(axis=1)
        meanLR = np.nanmean(data[:,i],axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if data is fracCorr:
            if stim=='targetOnly':
                firstValid = 0
            elif stim=='mask':
                firstValid = 0
            else:
                firstValid = 0
#            lbls = ('response rate not above chance','response rate above chance') if stim=='maskOnly' else (None,None)
#            ax.plot(xticks[:firstValid],mean[:firstValid],'o',mec=clr,mfc='none',label=lbls[0])
#            ax.plot(xticks[firstValid:-1],mean[firstValid:-1],'o',mec=clr,mfc=clr,label=lbls[1])
        else:
            firstValid = 0
        lbl = stimLbl if data is respRate else None
        ax.plot(xticks[firstValid:-1],mean[firstValid:-1],color=clr)
        ax.plot(xticks[-1],mean[-1],'o',color=clr,label=lbl)
        for x,m,s in zip(xticks[firstValid:],mean[firstValid:],sem[firstValid:]):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    if data is respRate:
        ax.legend()
    plt.tight_layout()


# unilateral opto
nexps = len(exps)
stimLabels = ('target left','target right','no stim')
optoSide = ('left','right','both','none')
ntrials = [{stim: [] for stim in stimLabels} for _ in range(nexps)]
respRate = [{stim: {respDir: [] for respDir in rewardDir} for stim in stimLabels} for _ in range(nexps)]
meanReacTime = copy.deepcopy(respRate)
for n,obj in enumerate(exps):
    validTrials = obj.engaged & (~obj.earlyMove)
    for stim in stimLabels:
        if stim=='no stim':
            stimTrials = validTrials & np.in1d(obj.trialType,('catch','catchOpto')) & np.isnan(obj.rewardDir)
        else:
            rd = 1 if stim=='target left' else -1
            stimTrials = validTrials & np.in1d(obj.trialType,('targetOnly','targetOnlyOpto')) & (obj.rewardDir==rd)
        for opSide in optoSide:
            if opSide=='left':
                optoTrials = stimTrials & obj.optoChan[:,0] & ~obj.optoChan[:,1]
            elif opSide=='right':
                optoTrials= stimTrials & ~obj.optoChan[:,0] & obj.optoChan[:,1]
            elif opSide=='both':
                optoTrials = stimTrials & obj.optoChan[:,0] & obj.optoChan[:,1]
            else:
                optoTrials = stimTrials & np.isnan(obj.optoOnset)
            for respDir in rewardDir:
                respTrials = optoTrials & (obj.responseDir==respDir)
                respRate[n][stim][respDir].append(respTrials.sum()/optoTrials.sum())
                meanReacTime[n][stim][respDir].append(np.nanmedian(obj.reactionTime[respTrials]))
                    
fig = plt.figure(figsize=(6,6))
xticks = np.arange(len(optoSide))
for i,stim in enumerate(stimLabels):    
    ax = fig.add_subplot(len(stimLabels),1,i+1)
    for respDir,clr,lbl in zip(rewardDir,'rb',('move right','move left')):
        d = [respRate[n][stim][respDir] for n in range(nexps)]
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(xticks,mean,'o-',color=clr,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(optoSide)
    ax.set_xlim([-0.25,len(optoSide)-0.75])
    if i==2:
        ax.set_xlabel('Optogenetic Stimulus Side')
        ax.set_yticks(np.arange(0,1,0.1))
        ax.set_ylim([0,0.2])
    else:
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0,1,0.2))
        ax.set_ylim([0,0.75])
    ax.set_ylabel('Probability')
    if i==0:
        ax.legend()
    ax.set_title(stim)
plt.tight_layout() 

                


