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
targetContrast = np.unique(exps[0].targetContrast)
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
xlim = [-0.05*targetContrast.max(),1.05*targetContrast.max()]

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
probGoRight = respRate.copy()
visRating = respRate.copy()
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
                    if hasattr(obj,'visRating'):
                        visRating[n,i,j] = obj.visRatingScore[trials].mean()
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
    for i,(data,ylim,ylabel) in enumerate(zip((respRate[n],fracCorr[n],medianReacTime[n]),((0,1.02),(0.4,1.02),None),('Response Rate','Fraction Correct','Median reaction time (ms)'))):
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
    
# stats
alpha = 0.05
for data,title in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):        
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
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(np.arange(len(xticklabels)))
    ax.set_yticklabels(xticklabels)
    ax.set_xlim([-0.5,len(xticklabels)-0.5])
    ax.set_ylim([-0.5,len(xticklabels)-0.5])
    ax.set_xlabel('Mask onset (ms)',fontsize=12)
    ax.set_ylabel('Mask onset (ms)',fontsize=12)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks(clim)
    cb.set_ticklabels(["{:.0e}".format(lim[0]),lim[1]])
    ax.set_title(title+' Comparisons (p value)')
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
ax = fig.add_subplot(1,1,1)
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
ax.set_xlabel('Reaction Time (ms)',fontsize=12)
ax.set_ylabel('Cumulative Probability',fontsize=12)
ax.legend(title='mask onset',loc='upper left',fontsize=8)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
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

# visibility rating
for n in range(len(exps)):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xlim,[0,0],'--',color='0.8')
#    for d,clr in zip(visRating[n],'rb'):
#        ax.plot(xticks,d,'o',color=clr)
#    meanLR = np.nansum(visRating[n]*respRate[n],axis=0)/np.sum(respRate[n],axis=0)
    meanLR = np.mean(visRating[n],axis=0)
    ax.plot(xticks,meanLR,'ko')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks([-1,0,1])
    ax.set_yticklabels(['Target not\nvisible','Unsure','Target\nvisible'])
    ax.set_xlim(xlim)
    ax.set_ylim([-1.02,1.02])
    ax.set_xlabel('Mask onset relative to target onset (ms)')
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
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','no stim'),'km')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.nansum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
            if stim=='targetOnly':
                meanLR[respAboveChancePval>=0.05] = np.nan
                meanLR[:,np.sum(~np.isnan(meanLR),axis=0)<2] = np.nan
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        lbl = stimLbl if data is respRate else None
        for d in meanLR:
            ax.plot(xticks,d,color=clr,alpha=0.2)
        ax.plot(xticks,mean,'o',color=clr,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Optogenetic light onset relative to target onset (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    if data is respRate:
        ax.legend(loc='upper left')
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
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(2*len(xticklabels)))
    ax.set_xticklabels(2*xticklabels)
    ax.set_yticks(np.arange(2*len(xticklabels)))
    ax.set_yticklabels(2*xticklabels)
    ax.set_xlim([-0.5,2*len(xticklabels)-0.5])
    ax.set_ylim([-0.5,2*len(xticklabels)-0.5])
    ax.set_xlabel('Optogenetic light onset (ms)',fontsize=12)
    ax.set_ylabel('Optogenetic light onset (ms)',fontsize=12)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks(clim)
    cb.set_ticklabels(["{:.0e}".format(lim[0]),lim[1]])
    ax.set_title(title+' Comparisons (p value)')
    plt.tight_layout()

# fraction correct vs response rate
rr = np.mean(respRate[:,0]-respRate[:,1,0,-1][:,None,None],axis=1)
fc,rt = [np.nansum(d[:,0]*respRate[:,0],axis=1)/np.nansum(respRate[:,0],axis=1) for d in (fracCorr,medianReacTime)]
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
    ax.set_xlabel('Response Rate Relative to Chance',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
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

for data,ylim,ylabel in zip((respRate,fracCorr,medianReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Median reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target only','target + mask','mask only','no stim'),'kbgm')):
        if data is respRate:
            meanLR = np.mean(data[:,i],axis=1)
        else:
            meanLR = np.nansum(data[:,i]*respRate[:,i],axis=1)/np.sum(respRate[:,i],axis=1)
            if stim in ('targetOnly','mask'):
                meanLR[respAboveChancePval[:,i]>=0.05] = np.nan
                meanLR[:,np.sum(~np.isnan(meanLR),axis=0)<2] = np.nan
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        lbl = stimLbl if data is respRate else None
        for d in meanLR:
            ax.plot(xticks,d,color=clr,alpha=0.2)
        ax.plot(xticks,mean,'o',color=clr,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([8,108])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Optogenetic light onset relative to target onset (ms)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    if data is respRate:
        ax.legend(loc='upper left')
    plt.tight_layout()
    
# stats
alpha = 0.05
for data,title in zip((respRate,fracCorr),('Response Rate','Fraction Correct')):       
    if data is respRate:
        meanLR = np.mean(data,axis=2)
    else:
        meanLR = np.nansum(data*respRate,axis=2)/np.sum(respRate,axis=2)
        meanLR[:,:2][respAboveChancePval>=0.05] = np.nan
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
    
    fig = plt.figure(facecolor='w',figsize=(8,7))
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.nanmin(pvalsCorr))),alpha)
    clim = np.log10(lim)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color=np.array((255, 251, 204))/255)
    im = ax.imshow(np.log10(pmatCorr),cmap=cmap,clim=clim)
    ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(4*len(xticklabels)))
    ax.set_xticklabels(4*xticklabels)
    ax.set_yticks(np.arange(4*len(xticklabels)))
    ax.set_yticklabels(4*xticklabels)
    ax.set_xlim([-0.5,4*len(xticklabels)-0.5])
    ax.set_ylim([-0.5,4*len(xticklabels)-0.5])
    ax.set_xlabel('Optogenetic light onset (ms)',fontsize=12)
    ax.set_ylabel('Optogenetic light onset (ms)',fontsize=12)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks(clim)
    cb.set_ticklabels(["{:.0e}".format(lim[0]),lim[1]])
    ax.set_title(title+' Comparisons (p value)')
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
                    
fig = plt.figure(figsize=(6,9))
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




