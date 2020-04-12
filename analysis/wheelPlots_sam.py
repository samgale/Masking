# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:07:06 2020

@author: svc_ccg
"""

import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis import import_data, create_df
from dataAnalysis import combine_files, combine_dfs
from behaviorAnalysis import get_files

df = create_df(import_data())

files = get_files('486634','masking_to_analyze') # 486634, 495786

df = combine_dfs(combine_files(files,'129','130','131')) # '129','130','131'  '212','213','214'


frameRate = 120
monWidthPix = 1920
wheelSpeedGain = 1200
quiescentFrames = 60
rewardThresh = 0.15*monWidthPix
quiescentThresh = 0.025*monWidthPix
initiationThresh = 0.5*np.pi/180*wheelSpeedGain


wheelPos = np.cumsum(np.stack([w[s:] for w,s in zip(df['WheelTrace'],df['stimStart']-df['trialStart']-quiescentFrames)]),axis=1)
wheelPos -= wheelPos[:,:quiescentFrames].mean(axis=1)[:,None]

soaList = np.unique(df['soa'])
soaLabels = ['no-go','mask only']+[str(soa) for soa in soaList if soa>0]
soaLabels[-1] = 'target only'
soaColors = np.ones((len(soaList),4))
soaColors[0,:3] = 0
soaColors[1,:3] = 0.5
soaColors[-np.sum(soaList>0):] = plt.cm.jet(np.linspace(0,1,np.sum(soaList>0)))

t = 1000/frameRate*(np.arange(wheelPos.shape[1])-quiescentFrames)
tinterp = np.arange(t[0],t[-1])


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ymin = 0
ymax = 0
timeLabels = ('initiation','movement','outcome')
responseTime = {time: {side: {resp: {soa: [] for soa in soaList} for resp in (-1,1)} for side in (-1,1)} for time in timeLabels}
for side in (-1,1):
    for resp in (-1,1):
        for soa,clr,lbl in zip(soaList,soaColors,soaLabels):
            if soa<0: # no-go
                trials = df['nogoMove']==side if resp>0 else []
            elif soa>0: # go
                trials = (df['rewDir']==side) & (df['resp']==resp)
            else: # mask only
                trials = df['maskOnlyMove']==side if resp>0 else []
            if any(trials):
                trials = trials & (df['soa']==soa)
                if any(trials):
                    wheel = wheelPos[trials]
                    trialsToUse = np.zeros(len(wheel),dtype=bool)
                    for i,w in enumerate(wheel):
                        winterp = np.interp(tinterp,t,w)
                        winterp *= side*resp
                        outcomeInd = np.where(winterp>rewardThresh)[0]
                        if len(outcomeInd)>0:
                            initInd = np.where(winterp[:outcomeInd[0]]<=initiationThresh)[0]
                            if len(initInd)>0:
                                initTime = tinterp[initInd][-1]+1
                                if initTime>100:
                                    outcomeTime = tinterp[outcomeInd[0]]
                                    responseTime['initiation'][side][resp][soa].append(initTime)
                                    responseTime['movement'][side][resp][soa].append(outcomeTime-initTime)
                                    responseTime['outcome'][side][resp][soa].append(outcomeTime)
                                    trialsToUse[i] = True
                    if resp==1 and soa>=0:
                        m = wheel[trialsToUse].mean(axis=0)/wheelSpeedGain*180/np.pi
                        ymin = min(ymin,m.min())
                        ymax = max(ymax,m.max())
                        leg = lbl if side>0 else None
                        ax.plot(t,m,color=clr,label=leg)
                    
ax.plot([200,200],[-2000,2000],'--',color='0.5',scaley=False)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,500])
ax.set_ylim([ymin*1.02,ymax*1.02])
ax.set_xlabel('Time from stimulus onset (ms)',fontsize=16)
ax.set_ylabel('$\Delta$ Wheel position (degrees)',fontsize=16)
ax.legend(loc='upper left')
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
for tlbl,alpha in zip(timeLabels[:2],(0.4,1)):
    for side,slbl in zip((-1,1),'LR'):
        for resp,rlbl,ls in zip((-1,1),('incorrect','correct'),('--','-')):
            rt = responseTime[tlbl][side][resp]
            meanRt = [np.mean(rt[soa]) for soa in rt]
            semRt = [np.std(rt[soa])/(len(rt[soa])**0.5) for soa in rt]
            clr = 'r' if side*resp==1 else 'b'
            lbl = tlbl+' '+slbl+' '+rlbl
            ax.plot(soaList[1:],meanRt[1:],linestyle=ls,color=clr,alpha=alpha,label=lbl)
            for soa,m,s in zip(soaList[1:],meanRt[1:],semRt[1:]):
                ax.plot([soa,soa],[m-s,m+s],color=clr,alpha=alpha)
ax.legend(loc='upper left',fontsize=8)    




# comments for Chelsea
    
# df['soa'] should be non-rounded floats

# rename ignoreTrials to something more descriptive like earlyMove or openLoopMove
    
# rename targetLength to targetDuration
    
# rename Qviolations to quiescentViolations
    
# rename wheelTrace to deltaWheel
