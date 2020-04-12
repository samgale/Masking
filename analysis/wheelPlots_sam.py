# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:07:06 2020

@author: svc_ccg
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from dataAnalysis import import_data, create_df
from dataAnalysis import combine_files, combine_dfs
from behaviorAnalysis import get_files


#
d = import_data()
df = create_df(d)

frameRate = round(1/np.median(d['frameIntervals'][:]))
monWidthPix = d['monSizePix'][0]
wheelSpeedGain = d['wheelSpeedGain'][()]
quiescentFrames = d['quiescentFrames'][()]
normRewardDistance = d['normRewardDistance'][()]

print(frameRate,monWidthPix,wheelSpeedGain,quiescentFrames,normRewardDistance)


#
files = get_files('495786','masking_to_analyze') # 486634, 495786
df = combine_dfs(combine_files(files,'129','130','131')) # '129','130','131'  '212','213','214'


#
param = 'targetDuration' # targetDuration  targetContrast  soa

if param=='targetDuration':
    paramName = 'Target Duration (ms)'
elif param=='targetContrast':
    paramName = 'Target Contrast (ms)'
elif param=='soa':
    paramName = 'SOA (ms)'

paramList = np.unique(df[param])
paramLabels = ['no go']+[str(int(round(prm))) for prm in paramList if prm>0]
paramColors = np.ones((len(paramList),4))
paramColors[0,:3] = 0
paramColors[-np.sum(paramList>0):] = plt.cm.jet(np.linspace(0,1,np.sum(paramList>0)))
if param=='soa':
    paramLabels[1] = 'mask only'
    paramLabels[-1] = 'target only'
    paramColors[1,:3] = 0.5

pmin = paramList.min()
pmax = paramList.max()
prange = pmax-pmin
plim = [pmin-prange*0.05,pmax+prange*0.05]
    
# plot perforamnce
performanceParams = ('responseRate','fractionCorrect')
performance = {param: {side: [] for side in (-1,1)} for param in performanceParams}
for prm,lbl in zip(paramList,paramLabels):
    for side in (-1,1):
        trials = df[param]==prm
        if lbl=='no go':
            performance['responseRate'][side].append(np.sum(df['nogoMove'][trials]==side)/trials.sum())
            performance['fractionCorrect'][side].append(np.nan)
        elif lbl=='mask only':
            performance['responseRate'][side].append(np.sum(df['maskOnlyMove'][trials]==side)/trials.sum())
            performance['fractionCorrect'].append(np.nan)
        else: # go
            trials = trials & (df['rewDir']==side)
            resp = df['resp'][trials]
            performance['responseRate'][side].append(np.sum(resp!=0)/trials.sum())
            performance['fractionCorrect'][side].append(np.sum(resp==1)/trials.sum())

for perf in performanceParams:
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(1,1,1)
    for side,clr in zip((-1,1),'br'):
        ax.plot(paramList[paramList<=0],np.array(performance[perf][side])[paramList<=0],clr+'o',mec=clr,mfc=clr)
        ax.plot(paramList[paramList>0],np.array(performance[perf][side])[paramList>0],clr+'-o',mec=clr,mfc=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(paramList)
    ax.set_xticklabels([p.replace(' ','\n') for p in paramLabels])
    ax.set_xlim(plim)
    ax.set_ylim([0,1.01])
    ax.set_xlabel(paramName,fontsize=16)
    ax.set_ylabel(perf,fontsize=16)
    plt.tight_layout()
        
# plot wheel position and reponse time
wheelPos = np.cumsum(np.stack([w[s:] for w,s in zip(df['deltaWheel'],df['stimStart']-df['trialStart']-quiescentFrames)]),axis=1)
wheelPos -= wheelPos[:,:quiescentFrames].mean(axis=1)[:,None]  

t = 1000/frameRate*(np.arange(wheelPos.shape[1])-quiescentFrames)
tinterp = np.arange(t[0],t[-1])

rewardThreshPix = normRewardDistance*monWidthPix
initiationThreshDeg = 0.5
initiationThreshPix = initiationThreshDeg*np.pi/180*wheelSpeedGain

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
showNogoRespTime = False  
xmin = 0
xmax = 500
xind = (t>=xmin) & (t<=xmax)
ymin = 0
ymax = 0
ax.plot([200,200],[-2000,2000],'--',color='0.5')
timeLabels = ('initiation','movement','outcome')
responseTime = {time: {side: {resp: {prm: [] for prm in paramList} for resp in (-1,1)} for side in (-1,1)} for time in timeLabels}
for side in (-1,1):
    for resp in (-1,1):
        for prm,clr,lbl in zip(paramList,paramColors,paramLabels):
            trials = df[param]==prm
            if lbl=='no go':
                trials = trials & (df['nogoMove']==side) if resp>0 and showNogoRespTime else []
            elif lbl=='mask only':
                trials = trials & (df['maskOnlyMove']==side) if resp>0 else []
            else: # go
                trials = trials & (df['rewDir']==side) & (df['resp']==resp)
            if any(trials):
                wheel = wheelPos[trials]
                trialsToUse = np.zeros(len(wheel),dtype=bool)
                for i,(w,openLoopTime) in enumerate(zip(wheel,df['openLoopFrames'][trials]*1000/frameRate)):
                    winterp = np.interp(tinterp,t,w)
                    winterp *= side*resp
                    openLoopSamples = int(openLoopTime)
                    outcomeInd = np.where(winterp[openLoopSamples:]-winterp[openLoopSamples]>rewardThreshPix)[0]
                    if len(outcomeInd)>0:
                        outcomeInd = outcomeInd[0]+openLoopSamples
                        initInd = np.where(winterp[:outcomeInd]<=initiationThreshPix)[0]
                        if len(initInd)>0:
                            initTime = tinterp[initInd][-1]+1
                            if initTime>100:
                                outcomeTime = tinterp[outcomeInd]
                                responseTime['initiation'][side][resp][prm].append(initTime)
                                responseTime['movement'][side][resp][prm].append(outcomeTime-initTime)
                                responseTime['outcome'][side][resp][prm].append(outcomeTime)
                                trialsToUse[i] = True
                if resp==1:
                    m = wheel[trialsToUse].mean(axis=0)/wheelSpeedGain*180/np.pi
                    ymin = min(ymin,m[xind].min())
                    ymax = max(ymax,m[xind].max())
                    leg = lbl if side>0 else None
                    ax.plot(t,m,color=clr,label=leg)                  
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin*1.02,ymax*1.02])
ax.set_xlabel('Time from stimulus onset (ms)',fontsize=16)
ax.set_ylabel('$\Delta$ Wheel position (degrees)',fontsize=16)
ax.legend(loc='upper left')
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
showIncorrect = False
for tlbl,fillMarker in zip(timeLabels[:2],(True,False)):
    for side,slbl in zip((-1,1),'LR'):
        for resp,rlbl,ls,alpha in zip((-1,1),('incorrect','correct'),('--','-'),(0.4,1)):
            if resp==1 or showIncorrect:
                rt = responseTime[tlbl][side][resp]
    #            rtMean = [np.mean(rt[prm]) for prm in rt]
    #            rtSem = [np.std(rt[prm])/(len(rt[prm])**0.5) for prm in rt]
                rtMedian = [np.median(rt[prm]) for prm in rt]
                rtMad = [scipy.stats.median_absolute_deviation(rt[prm]) for prm in rt]
    #            rtCI = [np.percentile([np.median(np.random.choice(rt[prm],len(rt[prm]),replace=True)) for _ in range(1000)],(5,95)) for prm in rt]
                clr = 'r' if side*resp==1 else 'b'
                mfc = clr if fillMarker else 'none'
                lbl = tlbl+' '+slbl+' '+rlbl
                ax.plot(paramList,rtMedian,'o',linestyle=ls,color=clr,mec=clr,mfc=mfc,label=lbl)
                for prm,m,s in zip(paramList,rtMedian,rtMad):
                    ax.plot([prm,prm],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
xticks,xticklabels = (paramList,paramLabels) if showNogoRespTime else (paramList[1:],paramLabels[1:])
ax.set_xticks(xticks)
ax.set_xticklabels([x.replace(' ','\n') for x in xticklabels])
ax.set_xlim(plim)
ax.set_xlabel(paramName,fontsize=16)
ax.set_ylabel('Time (ms)',fontsize=16)
loc = 'upper left' if param=='soa' else 'upper right'
ax.legend(loc=loc,fontsize=10)  
plt.tight_layout()



