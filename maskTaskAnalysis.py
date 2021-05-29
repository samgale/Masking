# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import numpy as np
import scipy.stats
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
targetSide = ('left','right')
rewardDir = (1,-1)


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
                    
xticks = list(targetFrames/frameRate*1000)
xticklabels = ['no\nstimulus'] + [str(int(round(x))) for x in xticks[1:]]
xlim = [-5,105]

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.25)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks[0],mean[0],'ko')
    ax.plot(xticks[1:],mean[1:],'ko')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Duration (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    plt.tight_layout() 


# target contrast
stimLabels = ('targetOnly','catch')
targetContrast = np.array([0,0.2,0.4,0.6,1])
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
        stimTrials = validTrials & (obj.trialType==stim)
        for j,tc in enumerate(targetContrast):
            tcTrials = stimTrials  & (obj.targetContrast==tc)
            if tcTrials.sum()>0:  
                for i,rd in enumerate(rewardDir):
                    trials = tcTrials & (obj.rewardDir==rd) if stim=='targetOnly' else tcTrials
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
                    
xticks = targetContrast
xticklabels = ['no\nstimulus'] + [str(x) for x in targetContrast[1:]]
xlim = [-0.05,1.05]

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.25)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks[0],mean[0],'ko')
    ax.plot(xticks[1:],mean[1:],'ko')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Target Contrast (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    plt.tight_layout()                  
    

# masking
stimLabels = ('maskOnly','mask','targetOnly','catch')
maskOnset = np.array([0,2,3,4,6])
ntrials = np.full((len(exps),len(rewardDir),len(maskOnset)+2),np.nan)
respRate = ntrials.copy()
fracCorr = respRate.copy()
probGoRight = ntrials.copy()
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
                    if stim in ('targetOnly','mask'):
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        medianReacTimeCorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                        medianReacTimeIncorrect[n,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][~correctTrials]
                    
#np.save(fileIO.saveFile('Save respRate',fileType='*.npy'),respRate)
#np.save(fileIO.saveFile('Save fracCorr',fileType='*.npy'),fracCorr)

xticks = list(maskOnset/frameRate*1000)+[67,83]
xticklabels = ['mask\nonly']+[str(int(round(x))) for x in xticks[1:-2]]+['target\nonly','no\nstimulus']
xlim = [-8,92]

# single experiment
for n in range(len(exps)):
    fig = plt.figure(figsize=(6,9))
    for i,(data,ylim,ylabel) in enumerate(zip((respRate[n],fracCorr[n],medianReacTime[n]),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)'))):
        ax = fig.add_subplot(3,1,i+1)
        for d,clr in zip(data,'rb'):
            ax.plot(xticks,d,'o',color=clr)
        if i==0:
            meanLR = np.mean(data,axis=0)
        else:
            meanLR = np.sum(data*respRate[n],axis=0)/np.sum(respRate[n],axis=0)
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
for data,ylim,ylabel in zip((respRate,fracCorr),((0,1),(0.4,1)),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if data is respRate:
        meanLR = np.mean(data,axis=1)
    else:
        meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    for d,clr in zip(meanLR,plt.cm.tab20(np.linspace(0,1,meanLR.shape[0]))):
        ax.plot(xticks,d,color=clr,alpha=0.25)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    ax.plot(xticks,mean,'ko')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    plt.tight_layout()
    
# spearman correlation of accuracy vs mask onset
rs = []
ps = []
for fc in np.sum(fracCorr*respRate,axis=1)/np.sum(respRate,axis=1):
    r,p = scipy.stats.spearmanr(np.arange(5),fc[1:6])
    rs.append(r)
    ps.append(p)
    
# performance by target side
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,1,len(maskOnset)-1))[::-1,:3]
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']
for data,ylabel in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1],'--',color='0.8')
    for i,clr,lbl in zip(range(1,6),clrs,lbls):
        ax.plot(data[:,0,i],data[:,1,i],'o',mec=clr,mfc=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,1.02])
    ax.set_ylim([0,1.02])
    ax.set_aspect('equal')
    ax.set_xlabel('Target Left '+ylabel,fontsize=12)
    ax.set_ylabel('Target Right '+ylabel,fontsize=12)
    if ylabel=='Response Rate':
        ax.legend(title='mask onset',loc='upper left',fontsize=8)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'--',color='0.8')
meanLR = np.sum(probGoRight*respRate,axis=1)/np.sum(respRate,axis=1)
for i,clr,lbl in zip(range(1,6),clrs,lbls):
    x = meanLR[:,0]
    y = meanLR[:,i]
    ax.plot(x,y,'o',mec=clr,mfc=clr)
    slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
    rx = np.array([min(x),max(x)])
    ax.plot(rx,slope*rx+yint,'-',color=clr,label=lbl+'(r='+str(round(rval,2))+', p='+str(round(pval,3))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Prob. Go Right (Mask Only)',fontsize=12)
ax.set_ylabel('Prob. Go Right (Target Either Side)',fontsize=12)
ax.legend(title='mask onset',loc='upper left',fontsize=8)
plt.tight_layout()
    
# reaction time on correct and incorrect trials
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for data,clr,lbl in zip((medianReacTimeCorrect,medianReacTimeIncorrect,medianReacTime),('k','0.8','k'),('correct','incorrect','other')):
    meanLR = np.sum(data*respRate,axis=1)/np.sum(respRate,axis=1)
    mean = np.nanmean(meanLR,axis=0)
    sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
    if lbl=='other':
        xt = [xticks[0],xticks[-1]]
        ax.plot(xt,mean[[0,-1]],'o',mec=clr,mfc='none',label=lbl)
        for x,m,s in zip(xt,mean[[0,-1]],sem[[0,-1]]):
            ax.plot([x,x],[m-s,m+s],'-',color=clr)
    else:
        ax.plot(xticks,mean,'o',color=clr,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],'-',color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_xlabel('Mask onset relative to target onset (ms)',fontsize=12)
ax.set_ylabel('Median reaction time (ms)',fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()

# fraction correct vs reaction time
binWidth = 50
bins = np.arange(0,650,binWidth)
rt = []
pc = []
for mo in [2,3,4,6,0]:
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
lbls = [lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]+['target only']
for r,n,clr,lbl in zip(rt,ntrials.sum(axis=(0,1))[1:6],clrs,lbls):
    s = np.sort(r)
    c = [np.sum(r<=i)/n for i in s]
    ax.plot(s,c,'-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False,labelsize=10)
ax.set_xlim([100,500])
ax.set_ylim([0,1.02])
ax.set_ylabel('Cumulative Probability',fontsize=12)
ax.legend(title='mask onset',loc='upper left',fontsize=8)

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

xticks = list((optoOnset[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']
                    
for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','no stim'),'km')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.sum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if data is fracCorr:
            if stim=='targetOnly':
                firstValid = 0 #2
            else:
                firstValid = 0
        else:
            firstValid = 0
        lbl = stimLbl if data is respRate else None
        for d in meanLR:
            ax.plot(xticks[firstValid:-1],d[firstValid:-1],color=clr,alpha=0.2)
            ax.plot(xticks[-1],d[-1],'o',mec=clr,mfc='none',alpha=0.2)
        ax.plot(xticks[firstValid:-1],mean[firstValid:-1],'o',color=clr)
        ax.plot(xticks[-1],mean[-1],'o',color=clr,label=lbl)
        for x,m,s in zip(xticks[firstValid:],mean[firstValid:],sem[firstValid:]):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    if data is respRate:
        ax.legend(loc='upper left')
    plt.tight_layout()

# fraction correct vs response rate
rr = np.mean(respRate[:,0]-respRate[:,1],axis=1)
fc,rt = [np.nansum(d[:,0]*respRate[:,0],axis=1)/np.sum(respRate[:,0],axis=1) for d in (fracCorr,medianReacTime)]
n = np.sum(ntrials[:,0]*respRate[:,0],axis=1)

bw = 0.2
bins = np.arange(-0.1,1.01,bw)
for data,ylim,ylabel in zip((fc,rt),((-0.02,1.02),None),('Fraction Correct','Reaction Time')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if ylabel=='Fraction Correct':
        ax.plot([bins[0],1],[0.5,0.5],'k:')
    ax.plot(rr,data,'o',mec='0.8',mfc='none')
    ind = np.digitize(rr,bins)
    for i,b in enumerate(bins[:-1]):
        bi = ind==i+1
        x = b+0.5*bw
        ntotal = n[bi].sum()
        m = np.nansum(data[bi]*n[bi]/ntotal)
        if ylabel=='Fraction Correct':
            s = [c/ntotal for c in scipy.stats.binom.interval(0.95,ntotal,m)]
#            ax.plot([x,x],[c/ntotal for c in scipy.stats.binom.interval(0.95,ntotal,0.5)],'r')
        else:
            s = np.nanstd(data[bi])/(bi.sum()**0.5)
            s = [m-s,m+s]
        ax.plot(x,m,'ko')
        ax.plot([x,x],s,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([-0.1,0.9])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Response Rate Above Chance',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    plt.tight_layout()


# opto masking
stimLabels = ('mask','targetOnly','maskOnly','catch')
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
        mo = 2 if stim in ('mask','maskOpto') else 0
        stimTrials = validTrials & np.in1d(obj.trialType,(stim,stim+'Opto')) & (obj.maskOnset==mo)
        for j,optoOn in enumerate(optoOnset):
            optoTrials = stimTrials & np.isnan(obj.optoOnset) if np.isnan(optoOn) else stimTrials & (obj.optoOnset==optoOn)
            for i,rd in enumerate(rewardDir):
                trials = optoTrials & (obj.rewardDir==rd) if stim in ('targetOnly','targetOnlyOpto','mask','maskOpto') else optoTrials
                ntrials[n,s,i,j] = trials.sum()
                respTrials = trials & (~np.isnan(obj.responseDir))
                respRate[n,s,i,j] = respTrials.sum()/trials.sum()
                medianReacTime[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials])
                reacTime[n][stim][rd][optoOn] = obj.reactionTime[respTrials]
                if stim in ('targetOnly','targetOnlyOpto','mask','maskOpto'):
                    correctTrials = obj.response[respTrials]==1
                    fracCorr[n,s,i,j] = correctTrials.sum()/respTrials.sum()
                    medianReacTimeCorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][correctTrials])
                    medianReacTimeIncorrect[n,s,i,j] = np.nanmedian(obj.reactionTime[respTrials][~correctTrials])
                    reacTimeCorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][correctTrials]
                    reacTimeIncorrect[n][stim][rd][optoOn] = obj.reactionTime[respTrials][~correctTrials]

xticks = list((optoOnset[:-1]-exps[0].frameDisplayLag)/frameRate*1000)+[100]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target + mask','target only','mask only','no stim'),'bckm')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.sum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if data is fracCorr:
            if stim=='targetOnly':
                firstValid = 3
            elif stim=='mask':
                firstValid = 2
            else:
                firstValid = 0
        else:
            firstValid = 0
        lbl = stimLbl if data is respRate else None
#        for d in meanLR:
#            ax.plot(xticks[firstValid:-1],d[firstValid:-1],color=clr,alpha=0.2)
#            ax.plot(xticks[-1],d[-1],'o',mec=clr,mfc='none',alpha=0.2)
        ax.plot(xticks[firstValid:-1],mean[firstValid:-1],'o',color=clr)
        ax.plot(xticks[-1],mean[-1],'o',color=clr,label=lbl)
        for x,m,s in zip(xticks[firstValid:],mean[firstValid:],sem[firstValid:]):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    if data is respRate:
        ax.legend(loc='upper left')
    plt.tight_layout()


# unilateral opto
nexps = len(exps)
stimLabels = ('target left','target right','no stim')
optoSide = ('left','right','both','no opto')
ntrials = [{stim: [] for stim in stimLabels} for _ in range(nexps)]
respRate = [{stim: {respDir: [] for respDir in rewardDir} for stim in stimLabels} for _ in range(nexps)]
medianReacTime = copy.deepcopy(respRate)
for n,obj in enumerate(exps):
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
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
                medianReacTime[n][stim][respDir].append(np.nanmedian(obj.reactionTime[respTrials]))
                    
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
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xlim([-0.25,len(optoSide)-0.75])
    if i==2:
        ax.set_xticklabels(optoSide)
        ax.set_xlabel('Optogenetic Stimulus Side',fontsize=12)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,1,0.2))
    ax.set_ylim([0,0.75])
    ax.set_ylabel('Probability',fontsize=12)
    if i==0:
        ax.legend()
    ax.set_title(stim)
plt.tight_layout()


# SAC figure for Shawn
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
meanLR = np.nanmean(fracCorr,axis=1)
mean = np.nanmean(meanLR,axis=0)
sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
ax.plot(xticks,mean,'ko',ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-')
ax.plot(xticks[1],optoMean[2],'co',ms=12)
ax.plot([xticks[1]]*2,[optoMean[2]-optoSem[2],optoMean[2]+optoSem[2]],color='c')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([8,75])
ax.set_ylim([0.5,1])
ax.set_xlabel('Mask onset relative to target onset (ms)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()

                


