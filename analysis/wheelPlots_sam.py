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

files = get_files('495786','masking_to_analyze') # 486634, 495786

df = combine_dfs(combine_files(files,'129','130','131')) # '129','130','131'  '212','213','214'

wheelSpeedGain = 1200

quiescentFrames = 60

frameRate = 120

wheel = np.cumsum(np.stack([w[s:] for w,s in zip(df['WheelTrace'],df['stimStart']-df['trialStart']-quiescentFrames)]),axis=1)
wheel -= wheel[:,:quiescentFrames].mean(axis=1)[:,None]

soaList = np.unique(df['soa'])
soaLabels = ['no-go','mask only']+[str(soa) for soa in soaList if soa>0]
soaLabels[-1] = 'target only'
soaColors = np.ones((len(soaList),4))
soaColors[0,:3] = 0.5
soaColors[1,:3] = 0
soaColors[-np.sum(soaList>0):] = plt.cm.jet(np.linspace(0,1,np.sum(soaList>0)))

t = 1000/frameRate*(np.arange(wheel.shape[1])-quiescentFrames)

plotNogo = False

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ymin = 0
ymax = 0
for soa,clr,lbl in zip(soaList,soaColors,soaLabels):
    for side in (-1,1):
        if soa<0: # no-go
            trials = df['nogoMove']==side if plotNogo else []
        elif soa>0: # go
            trials = (df['resp']==1) & (df['rewDir']==side)
        else: # mask only
            trials = df['maskOnlyMove']==side
        if any(trials):
            leg = lbl if side>0 else None
            m = np.mean(wheel[trials & (df['soa']==soa)],axis=0)/wheelSpeedGain*180/np.pi
            ymin = min(ymin,m.min())
            ymax = max(ymax,m.max())
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




# comments for Chelsea
    
# df['soa'] should be non-rounded floats

# rename ignoreTrials to something more descriptive like earlyMove or openLoopMove
    
# rename targetLength to targetDuration
    
# rename Qviolations to quiescentViolations
    
# rename wheelTrace to deltaWheel
