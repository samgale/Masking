# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import math
import os
import pickle
import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.stats
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import fileIO
from maskTaskAnalysisUtils import MaskTaskData,getPsth,loadDatData



obj = MaskTaskData()

obj.loadEphysData(led=False)

obj.loadKilosortData()

obj.loadBehavData()

obj.loadRFData()

obj.saveToHdf5()

obj.loadFromHdf5()      


exps = []
for f in fileIO.getFiles('choose experiments',fileType='*.hdf5'):
    obj = MaskTaskData()
    obj.loadFromHdf5(f)
    obj.loadBehavData(obj.behavDataPath)
    obj.loadKilosortData(os.path.join(os.path.dirname(obj.datFilePath),'kilosort_filtered'))
    exps.append(obj)
    

unitPos = []
peakToTrough = []
for obj in exps:
    units = obj.goodUnits
    unitPos.extend([obj.units[u]['position'][1]/1000 for u in units])
    peakToTrough.extend([obj.units[u]['peakToTrough'] for u in units])
unitPos = np.array(unitPos)
peakToTrough = np.array(peakToTrough)

fsThresh = 0.4
fs = peakToTrough < fsThresh


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bw = 0.1
bins = np.arange(0,peakToTrough.max()+bw,bw)
h = np.histogram(peakToTrough,bins=bins)[0]
ax.bar(x=bins[:-1]+bw/2,height=h,width=bw,color='k')
ymax = plt.get(ax,'ylim')[1]
ax.plot([fsThresh]*2,[0,ymax],'--',color='0.5')
for x,lbl in zip((0.2,1),('FS','RS')):
    ax.text(x,ymax,lbl,color='0.5',fontsize=14,ha='center',va='top')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1.6])
ax.set_xlabel('Spike Peak-to-Trough Duration (ms)',fontsize=16)
ax.set_ylabel('# Units',fontsize=16)
plt.tight_layout()

fpRate = np.array([obj.units[u]['fpRate'] for obj in exps for u in obj.sortedUnits if obj.units[u]['label']!='noise' and len(obj.units[u]['samples'])/(obj.totalSamples/obj.sampleRate)>0.1])
fpRate[fpRate>1] = 1
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bw = 0.1
bins = np.arange(0,1.05,bw)
h = np.histogram(fpRate,bins=bins)[0]
ax.bar(x=bins[:-1]+bw/2,height=h,width=bw,color='k')
ymax = plt.get(ax,'ylim')[1]
ax.plot([0.5]*2,[0,ymax],'--',color='0.5')
for x,lbl in zip((0.25,0.75),('accepted','rejected')):
    ax.text(x,ymax,lbl,color='0.5',fontsize=14,ha='center',va='top')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_xlabel('False-Positive Rate',fontsize=16)
ax.set_ylabel('# Units',fontsize=16)
plt.tight_layout()


# plot response to visual stimuli without opto
trialTime = [(obj.openLoopFrames+obj.responseWindowFrames)/obj.frameRate for obj in exps]
assert(all([t==trialTime[0] for t in trialTime]))
trialTime = trialTime[0]

frameRate = 120
binSize = 1/frameRate
activeThresh = 0.1 # spikes/s
respThresh = 5 # stdev
stimLabels = ('maskOnly','mask','targetOnly','catch')
hemiLabels = ('contra','ipsi')
rewardDir = (-1,1)
behavRespLabels = ('all','go','nogo')
preTime = 0.5
postTime = 1
windowDur = preTime+trialTime+postTime

ntrials = {stim: {hemi: {resp: {} for resp in behavRespLabels} for hemi in hemiLabels} for stim in stimLabels}
trialPsth = copy.deepcopy(ntrials)
psth = copy.deepcopy(ntrials)
hasSpikes = copy.deepcopy(psth)
hasResp = copy.deepcopy(psth)
peakResp = copy.deepcopy(psth)
timeToFirstSpike = copy.deepcopy(psth)
for obj in exps:
    ephysHemi = hemiLabels if hasattr(obj,'hemi') and obj.hemi=='right' else hemiLabels[::-1]
    validTrials = ~obj.longFrameTrials
    for stim in stimLabels:
        for rd,hemi in zip((1,-1),ephysHemi):
            stimTrials = obj.trialType==stim
            if stim in ('targetOnly','mask'):
                stimTrials = stimTrials & (obj.rewardDir==rd)
            for resp in behavRespLabels:
                if resp=='all':
                    respTrials = np.ones(obj.ntrials,dtype=bool)
                else:
                    respTrials = ~np.isnan(obj.responseDir) if resp=='go' else np.isnan(obj.responseDir)
                for mo in np.unique(obj.maskOnset[stimTrials]):
                    moTrials = obj.maskOnset==mo
                    trials = validTrials & stimTrials & respTrials & (obj.maskOnset==mo)
                    if mo not in ntrials[stim][hemi][resp]:
                        ntrials[stim][hemi][resp][mo] = 0
                        trialPsth[stim][hemi][resp][mo] = []
                        psth[stim][hemi][resp][mo] = []
                        hasSpikes[stim][hemi][resp][mo] = []
                        hasResp[stim][hemi][resp][mo] = []
                        peakResp[stim][hemi][resp][mo] = []
                        timeToFirstSpike[stim][hemi][resp][mo] = []
                    ntrials[stim][hemi][resp][mo] += trials.sum()
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for u in obj.goodUnits:
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPsth(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
                        trialPsth[stim][hemi][resp][mo].append(p)
                        m = p.mean(axis=0)
                        t -= preTime
                        psth[stim][hemi][resp][mo].append(m)
                        analysisWindow = (t>0.04) & (t<0.1)
                        hasSpikes[stim][hemi][resp][mo].append(m[analysisWindow].mean() > activeThresh)
                        b = p[:,t<0].mean(axis=1)
                        r = p[:,analysisWindow].mean(axis=1)
                        peak = (m-b.mean())[analysisWindow].max()
                        pval = 1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] 
                        hasResp[stim][hemi][resp][mo].append(peak>respThresh*m[t<0].std() and pval<0.05)
                        peakResp[stim][hemi][resp][mo].append(peak)
                        lat = []
                        for st in startTimes:
                            firstSpike = np.where((spikeTimes > st+0.03) & (spikeTimes < st+0.15))[0]
                            if len(firstSpike)>0:
                                lat.append(spikeTimes[firstSpike[0]]-st)
                            else:
                                lat.append(np.nan)
                        timeToFirstSpike[stim][hemi][resp][mo].append(np.nanmedian(lat)*1000)

activeUnits = np.array(hasSpikes['targetOnly']['contra']['all'][0]) | np.array(hasSpikes['maskOnly']['contra']['all'][0])
targetRespUnits = np.array(hasResp['targetOnly']['contra']['all'][0])
maskRespUnits = np.array(hasResp['maskOnly']['contra']['all'][0])
respUnits = targetRespUnits | maskRespUnits

xlim = [-0.1,0.65]
for units in (respUnits,):
    axs = []
    ymin = ymax = 0
    for resp in ('all',): #behavRespLabels:
        fig = plt.figure(figsize=(10,5))
        fig.text(0.5,0.99,'n='+str(units.sum())+' units)',ha='center',va='top',fontsize=12)
        for i,hemi in enumerate(hemiLabels):
            ax = fig.add_subplot(1,2,i+1)
            axs.append(ax)
            for stim,clr in zip(stimLabels,('0.5','r','k','m')):
                mskOn = psth[stim][hemi][resp].keys()
                if stim=='mask' and len(mskOn)>1:
                    cmap = np.ones((len(mskOn),3))
                    cint = 1/len(mskOn)
                    cmap[:,1:] = np.arange(0,1.01-cint,cint)[:,None]
                else:
                    cmap = [clr]
                for mo,c in zip(mskOn,cmap):
                    p = np.array(psth[stim][hemi][resp][mo])[units]
                    m = np.mean(p,axis=0)
                    s = np.std(p,axis=0)/(len(p)**0.5)
                    lbl = 'target+mask, SOA '+str(round(1000*mo/frameRate,1))+' ms' if stim=='mask' else stim
                    rlbl = '' if resp=='all' else ' '+resp
                    lbl += ' ('+str(ntrials[stim][hemi][resp][mo])+rlbl+' trials)'
                    ax.plot(t,m,color=c,label=lbl)
#                    ax.fill_between(t,m+s,m-s,color=c,alpha=0.25)
                    ymin = min(ymin,np.min(m[(t>=xlim[0]) & (t<=xlim[1])]))
                    ymax = max(ymax,np.max(m[(t>=xlim[0]) & (t<=xlim[1])]))
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from stimulus onset (s)')
            if i==0:
                ax.set_ylabel('Response (spikes/s)')
            ax.legend(loc='upper left',frameon=False,fontsize=8)
            ax.set_title('target '+hemi)
    for ax in axs:
        ax.set_ylim([1.05*ymin,1.05*ymax])
        

# peak response and latency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ymax = 0
for stim,clr,lbl in zip(('targetOnly','maskOnly'),'kg',('Target Only','Mask Only')):
    p = np.array(psth[stim]['contra']['all'][0])[respUnits]
    m = p.mean(axis=0)
    s = p.std(axis=0)/(len(p)**0.5)
    ax.plot(t*1000,m,color=clr,label=lbl)
    ax.fill_between(t*1000,m+s,m-s,color=clr,alpha=0.25)
    ymax = max(ymax,np.max(m+s))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,150])
ax.set_ylim([0,42])
ax.set_xlabel('Time From Stimulus Onset (ms)',fontsize=16)
ax.set_ylabel('Spikes/s',fontsize=16)
ax.legend(loc='upper right',fontsize=14)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for stim,clr in zip(('targetOnly','maskOnly'),'kg'):
    d = np.array(timeToFirstSpike[stim]['contra']['all'][0])[respUnits]
    s = np.sort(d)
    c = [np.sum(s<=i)/len(s) for i in s]
    ax.plot(s,c,color=clr,lw=2,label=stim)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,150])
ax.set_ylim([0,1.02])
ax.set_xlabel('Median Time To First Spike (ms)',fontsize=16)
ax.set_ylabel('Cumulative Probability',fontsize=16)
#ax.legend(loc='upper right',fontsize=14)
plt.tight_layout()

for data,amax,albl in zip((peakResp,timeToFirstSpike),(200,150),('Peak Response (spikes/s)','Median Time To First Spike (ms)')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ymax = 0
    ax.plot([0,1000],[0,1000],'--',color='0.8')
    for cellType,clr,lbl in zip((~fs,fs),'km',('RS','FS')):
        d = [np.array(data[stim]['contra']['all'][0])[respUnits & cellType] for stim in ('targetOnly','maskOnly')]
        ax.plot(d[0],d[1],'o',color=clr,label=lbl)
        ymax = max(ymax,np.nanmax(d))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    aticks = np.arange(0,amax+1,50)
    ax.set_xticks(aticks)
    ax.set_yticks(aticks)
    ax.set_xlim([0,amax])
    ax.set_ylim([0,amax])
    ax.set_aspect('equal')
    ax.set_xlabel('Target',fontsize=16)
    ax.set_ylabel('Mask',fontsize=16)
    ax.set_title(albl,fontsize=16)
    if 'Peak Response' in albl:
        ax.legend(loc='lower right',fontsize=14)
    plt.tight_layout()

for data,lbl in zip((peakResp,timeToFirstSpike),('peak response','time to first spike')):   
    d = [np.array(data[stim]['contra']['all'][0])[respUnits] for stim in ('targetOnly','maskOnly')]
    m = [np.nanmedian(i) for i in d]
    pval = scipy.stats.ranksums(d[0],d[1])[1]
    print(lbl,('target','mask'),m,pval)
    
for data,lbl in zip((peakResp,timeToFirstSpike),('peak response','time to first spike')): 
    for stim in ('targetOnly','maskOnly'):
        d = [np.array(data[stim]['contra']['all'][0])[respUnits & ct] for ct in (~fs,fs)]
        m = [np.nanmedian(i) for i in d]
        pval = scipy.stats.ranksums(d[0],d[1])[1]
        print(lbl,stim,('RS','FS'),m,pval)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for cellType,clr,lbl in zip((~fs,fs),'km',('RS','FS')):
    d = [np.array(data['maskOnly']['contra']['all'][0])[respUnits & cellType] for data in (timeToFirstSpike,peakResp)]
    ax.plot(d[0],d[1],'o',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlabel('Time To First Spike (ms)',fontsize=16)
ax.set_ylabel('Peak Response To Mask (spikes/s)',fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()


# compare contra, ipsi, and predicted response to target + mask
for units in (respUnits,): #targetRespUnits,maskRespUnits & ~targetRespUnits):
    units = units & ~fs
    fig = plt.figure(figsize=(4,6))
    axs = []
    ymin = 0
    ymax = 0
    i = 0
    target = np.array(psth['targetOnly']['contra']['all'][0])[units]
    for stim in ('targetOnly','mask','maskOnly'):
        for mo in psth[stim]['contra']['all']:
            ax = fig.add_subplot(6,1,i+1)
            
            for hemi,clr in zip(hemiLabels,'rb'):
                p = np.array(psth[stim][hemi]['all'][mo])[units]
                b = p-p[:,t<0].mean(axis=1)[:,None]
                m = np.mean(b,axis=0)
                s = np.std(b,axis=0)/(len(p)**0.5)
                if stim=='targetOnly':
                    lbl = hemi+'lateral target'
                elif stim=='maskOnly':
                    if hemi=='contra':
                        clr = 'k'
                        lbl = 'no target'
                    else:
                        continue
                else:
                    lbl = None
                ax.plot(t*1000,m,color=clr,label=lbl)
                ax.fill_between(t*1000,m+s,m-s,color=clr,alpha=0.25)
                ymin = min(ymin,np.min(m-s))
                ymax = max(ymax,np.max(m+s))
            
            if stim=='mask':
                mask = np.array(psth['mask']['ipsi']['all'][mo])[units]
                p = target+mask
                b = p-p[:,t<0].mean(axis=1)[:,None]
                m = np.mean(b,axis=0)
                s = np.std(b,axis=0)/(len(p)**0.5)
                ax.plot(t*1000,m,ls='--',color='0.5',label='linear sum\ntarget + mask')
                ax.fill_between(t*1000,m+s,m-s,color='0.5',alpha=0.25)
                ymin = min(ymin,np.min(m-s))
                ymax = max(ymax,np.max(m+s))
            if i==5:
                ax.set_xlabel('Time From Stimulus Onset (ms)',fontsize=12)
            else:
                ax.set_xticklabels([])
            if i==0:
                ax.set_ylabel('Spikes/s',fontsize=12)
            title = stim
            if stim=='targetOnly':
                title = 'target only'
            elif stim=='maskOnly':
                title = 'mask only'
            elif stim=='mask':
                title = title+' onset\n'+str(int(round(mo/120*1000)))+' ms'
            ax.text(0.15,0.99,title,transform=ax.transAxes,color='k',fontsize=10,ha='center',va='top')
            if i in (0,1,5):
                ax.legend(fontsize=8)
            axs.append(ax)
            i += 1
    for ax in axs:
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([0,200])
        ax.set_ylim([1.05*ymin,1.05*ymax])
    plt.tight_layout()


# cumulative spike count and excess contralateral spikes
maskOnset = np.unique(exps[0].maskOnset[~np.isnan(exps[0].maskOnset)])
maskOnsetTicks = np.concatenate((maskOnset,(8,10)))/frameRate*1000
maskOnsetLabels = ['mask only']+[str(int(round(onset)))+' ms' for onset in maskOnsetTicks[1:-2]] + ['target only','no stim']
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]

units = respUnits & ~fs
analysisWindow = (t>0.033) & (t<0.2)

cumContra = []
cumIpsi =[]
for stim in stimLabels:
    for mo in maskOnset:
        if stim!='mask' and mo>0 or stim=='mask' and mo==0:
            continue
        cumSpikes = []
        for hemi in ('contra','ipsi'):
            p = np.array(psth[stim][hemi]['all'][mo])[units]
            b = p-p[:,t<0].mean(axis=1)[:,None]
            cumSpikes.append(np.cumsum(b[:,analysisWindow],axis=1)*binSize)
        cumContra.append(cumSpikes[0])
        cumIpsi.append(cumSpikes[1])

for i,lbl in enumerate(maskOnsetLabels): 
    d = cumContra[i].max(axis=1)      
    print(lbl,np.median(d),np.min(d),np.max(d))

d = np.array(cumContra)[1:5].max(axis=2)
print(np.median(d),np.min(d),np.max(d))

for resp,hemi in zip((cumContra,cumIpsi),('Contralateral','Ipsilateral')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip(resp[1:-1],clrs,maskOnsetLabels[1:-1]):
        m = d.mean(axis=0)
        s = d.std(axis=0)/(len(d)**0.5)
        ax.plot(t[analysisWindow]*1000,m,color=clr,label=lbl)
        ax.fill_between(t[analysisWindow]*1000,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([33,200])
    ax.set_yticks([0,0.5,1,1.5])
    ax.set_ylim([-0.05,1.5])
    ax.set_xlabel('Time Relative to Target Onset (ms)',fontsize=16)
    ax.set_ylabel('Cumulative Spikes Per Neuron',fontsize=16)
#    ax.set_title(hemi+' Target',fontsize=14)
    if hemi=='Contralateral':
        leg = ax.legend(title='mask onset',loc='upper left',fontsize=12)
        plt.setp(leg.get_title(),fontsize=12)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for contra,ipsi,clr,lbl in zip(cumContra[1:-1],cumIpsi[1:-1],clrs,maskOnsetLabels[1:-1]):
    d = contra-ipsi
    m = d.mean(axis=0)
    s = d.std(axis=0)/(len(d)**0.5)
    ax.plot(t[analysisWindow]*1000,m,color=clr,label=lbl)
    ax.fill_between(t[analysisWindow]*1000,m+s,m-s,color=clr,alpha=0.25)
ax.plot([33,200],[0,0],'k--')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([33,200])
ax.set_xlabel('Time Relative to Target Onset (ms)',fontsize=16)
ax.set_ylabel('Cumulative Spike Count Difference',fontsize=16)
ax.set_title('Contralateral - Ipsilateral',fontsize=14)
plt.tight_layout()


# save psth
units = respUnits & ~fs
popPsth = {stim: {hemi: {} for hemi in hemiLabels} for stim in stimLabels}
for stim in stimLabels:
    for hemi in hemiLabels:
        p = psth[stim][hemi]['all']
        for mo in p.keys():
            popPsth[stim][hemi][mo] = np.array(p[mo])[units]
popPsth['t'] = t
            
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(popPsth,open(pkl,'wb'))


# behavior analysis
rtBins = ((100,200),(200,400))
behavOutcomeLabels = ('all','correct','incorrect','no resp') + rtBins
behavTrialPsth = {stim: {hemi: {resp: {} for resp in behavOutcomeLabels} for hemi in hemiLabels} for stim in ('targetOnly','mask')}
behavPsth = copy.deepcopy(behavTrialPsth)
reacTime = copy.deepcopy(behavTrialPsth)
for i,obj in enumerate(exps):
    ephysHemi = hemiLabels if hasattr(obj,'hemi') and obj.hemi=='right' else hemiLabels[::-1]
    validTrials = (~obj.longFrameTrials) & obj.engaged & (~obj.earlyMove)
    for stim in ('targetOnly','mask'):
        for rd,hemi in zip((1,-1),ephysHemi):
            stimTrials = (obj.trialType==stim) & (obj.rewardDir==rd)
            for resp,respLbl in zip(('all',1,-1,0)+rtBins,behavOutcomeLabels):
                respTrials = obj.response==resp if resp in (-1,0,1) else np.ones(len(stimTrials),dtype=bool)
                for mo in np.unique(obj.maskOnset[stimTrials]):
                    moTrials = obj.maskOnset==mo
                    trials = validTrials & stimTrials & respTrials & (obj.maskOnset==mo) & np.isnan(obj.optoOnset)
                    if resp in rtBins:
                        trials = trials & (obj.reactionTime>=resp[0]) & (obj.reactionTime < resp[1])
                    if mo not in behavTrialPsth[stim][hemi][respLbl]:
                        behavTrialPsth[stim][hemi][respLbl][mo] = []
                        behavPsth[stim][hemi][respLbl][mo] = []
                        reacTime[stim][hemi][respLbl][mo] = []
                    reacTime[stim][hemi][respLbl][mo].append(obj.reactionTime[trials])
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate - preTime
                    for u in obj.goodUnits:
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPsth(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
                        behavTrialPsth[stim][hemi][respLbl][mo].append(p)
                        behavPsth[stim][hemi][respLbl][mo].append(p.mean(axis=0))                       
t -= preTime                       
                        
expInd = np.array([i for i,obj in enumerate(exps) for _ in enumerate(obj.goodUnits)])

isOptoExpUnit = np.array([np.any(~np.isnan(obj.optoOnset)) for obj in exps for unit in obj.goodUnits])

behavUnits = respUnits & ~isOptoExpUnit

maskOnset = (2,3,4,6,0)

for hemi in hemiLabels: 
    for mo,moLbl in zip(maskOnset,maskOnsetLabels[1:-1]):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        xmax = 0
        for resp,clr in zip(('correct','incorrect'),('g','m')):
            stim = 'mask' if mo>0 else 'targetOnly'
            p = np.array(behavPsth[stim][hemi][resp][mo])[behavUnits]
            b = p-p[:,t<0].mean(axis=1)[:,None]
            m = np.nanmean(b,axis=0)
            s = np.nanstd(b,axis=0)/(b.shape[0]**0.5)
            ax.plot(t,m,color=clr,lw=2,label=resp)
            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,0.2])
        ax.set_xlabel('Time (s)',fontsize=14)
        ax.set_ylabel('Spikes/s',fontsize=14)
        ax.set_title(hemi+' '+moLbl,fontsize=14)
        ax.legend()


analysisWindow = (t>0.035) & (t<0.075)
maskOnsetTicks = np.concatenate(((0,2,3,4,6),(8,10)))/frameRate*1000
maskOnsetLabels = ['mask only']+[str(int(round(onset)))+' ms' for onset in maskOnsetTicks[1:-2]] + ['target only','no stim']
clrs = np.zeros((len(maskOnset),3))
clrs[:-1] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [-0.1,0.8]
ax.plot(alim,alim,'--',color='0.8')
for mo,clr,moLbl in zip(maskOnset,clrs,maskOnsetLabels[1:-1]):
    stim = 'mask' if mo>0 else 'targetOnly'
    for hemi,mfc in zip(hemiLabels,(clr,'none')):
        m = []
        s = []
        for resp in ('correct','incorrect'):
            p = np.array(behavPsth[stim][hemi][resp][mo])[behavUnits]
            b = p-p[:,t<0].mean(axis=1)[:,None]
            r = b[:,analysisWindow].sum(axis=1) * binSize
            m.append(np.nanmean(r))
            s.append(np.nanstd(r)/len(r)**0.5)
        my,mx = m
        sy,sx = s
        lbl = hemi+' '+moLbl if stim=='targetOnly' else moLbl+' ('+hemi+' target)'
        ax.plot(mx,my,'o',mec=clr,mfc=mfc,label=lbl)
        ax.plot([mx-sx,mx+sx],[my,my],color=clr)
        ax.plot([mx,mx],[my-sy,my+sy],color=clr)    
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(0,1.2,0.2))
ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Spikes on incorrect trials',fontsize=14)
ax.set_ylabel('Spikes on correct trials',fontsize=14)
leg = ax.legend(title='mask onset',bbox_to_anchor=(0.8,0.8),fontsize=10)
plt.setp(leg.get_title(),fontsize=10)
plt.tight_layout()

  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [150,400]
ax.plot(alim,alim,'--',color='0.8')    
for mo,clr,lbl in zip(maskOnset,clrs,maskOnsetLabels[1:-1]):
    stim = 'mask' if mo>0 else 'targetOnly'
    m = []
    s = []
    for resp in ('correct','incorrect'):
        rt = [np.nanmedian(rt) for rt in np.concatenate([reacTime[stim][hemi][resp][mo] for hemi in hemiLabels])]
        m.append(np.nanmean(rt))
        s.append(np.nanstd(rt)/(len(rt)**0.5))
    my,mx = m
    sy,sx = s
    ax.plot(mx,my,'o',mec=clr,mfc=clr,label=lbl)
    ax.plot([mx-sx,mx+sx],[my,my],color=clr)
    ax.plot([mx,mx],[my-sy,my+sy],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Reaction time on incorrect trials (ms)',fontsize=14)
ax.set_ylabel('Reaction time on correct trials (ms)',fontsize=14)
leg = ax.legend(title='mask onset',loc='upper left',fontsize=12)
plt.setp(leg.get_title(),fontsize=12)
plt.tight_layout()  

    
for mo,title in zip(maskOnset,maskOnsetLabels[1:-1]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xmax = 0
    for resp,clr in zip(rtBins,plt.cm.jet(np.linspace(0,1,len(rtBins)))):
        stim = 'mask' if mo>0 else 'targetOnly'
        p = np.array(behavPsth[stim]['contra'][resp][mo])[behavUnits]
        b = p-p[:,(t<0) & (t>-0.15)].mean(axis=1)[:,None]
        m = np.nanmean(b,axis=0)
        s = np.nanstd(b,axis=0)/(b.shape[0]**0.5)
        ax.plot(t,m,color=clr,lw=2,label=resp)
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,0.2])
    ax.set_xlabel('Time (s)',fontsize=14)
    ax.set_ylabel('Spikes/s',fontsize=14)
    ax.set_title(title,fontsize=14)
    ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for mo,clr,lbl in zip(maskOnset,clrs,maskOnsetLabels[1:-1]):
    stim = 'mask' if mo>0 else 'targetOnly'
    mean = []
    sem = []
    xdata = []
    for rt in rtBins:
        p = np.array(behavPsth[stim]['contra'][rt][mo])[behavUnits]
        b = p-p[:,t<0].mean(axis=1)[:,None]
        r = b[:,analysisWindow].sum(axis=1) * binSize
        mean.append(np.nanmean(r))
        sem.append(np.nanstd(r)/(len(r)**0.5))
        xdata.append(sum(rt)/2)
    ax.plot(xdata,mean,'o-',color=clr,label=lbl)
    for x,m,s in zip(xdata,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlabel('Reaction time (ms)',fontsize=14)
ax.set_ylabel('Spikes',fontsize=14)
ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [-0.1,0.8]
ax.plot(alim,alim,'--',color='0.8')
for mo,clr,moLbl in zip(maskOnset,clrs,maskOnsetLabels[1:-1]):
    stim = 'mask' if mo>0 else 'targetOnly'
    for hemi,mfc in zip(hemiLabels,(clr,'none')):
        m = []
        s = []
        for rt in rtBins:
            p = np.array(behavPsth[stim][hemi][rt][mo])[behavUnits]
            b = p-p[:,t<0].mean(axis=1)[:,None]
            r = b[:,analysisWindow].sum(axis=1) * binSize
            m.append(np.nanmean(r))
            s.append(np.nanstd(r)/(len(r)**0.5))
        my,mx = m
        sy,sx = s
        lbl = hemi+' '+moLbl if stim=='targetOnly' else moLbl+' ('+hemi+' target)'
        ax.plot(mx,my,'o',mec=clr,mfc=mfc,label=lbl)
        ax.plot([mx-sx,mx+sx],[my,my],color=clr)
        ax.plot([mx,mx],[my-sy,my+sy],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(0,1.2,0.2))
ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Spikes on slow-reaction trials',fontsize=14)
ax.set_ylabel('Spikes on fast-reaction trials',fontsize=14)
leg = ax.legend(title='mask onset',bbox_to_anchor=(0.8,0.8),fontsize=10)
plt.setp(leg.get_title(),fontsize=10)
plt.tight_layout()



# decoding
def getDecoderResult(var,varLabels,units,dPsth,trainInd,testInd,numTrials,randomizeTrials,analysisWindow,dataType,offsets,maskOnset):
    # X = features, y = decoding label, m = mask onset label
    X = {trialSet: [] for trialSet in ('train','test')}
    y = [] 
    m = []
    for trialSet,trialInd in zip(('train','test'),(trainInd,testInd)):
        for i,u in enumerate(units):
            X[trialSet].append([])
            for vind,vlbl in enumerate(varLabels):
                for mo in maskOnset:
                    stim = 'targetOnly' if mo==0 else 'mask'
                    p = dPsth[stim][vlbl]['all'] if var=='side' else dPsth[stim]['contra'][vlbl]
                    ind = trialInd[vlbl][mo][u]
                    trials =  np.random.choice(ind,numTrials,replace=True) if randomizeTrials else ind[:numTrials]
                    for trial in trials:
                        X[trialSet][-1].append(p[mo][unitInd[u]][trial][analysisWindow])
                        if i==0 and trialSet=='train':
                            y.append(vind)
                            m.append(mo)
        X[trialSet] = np.concatenate(X[trialSet],axis=1)
    y = np.array(y)
    m = np.array(m)
    
    decoder = LinearSVC(C=1.0,max_iter=1e4)
    trainScore = np.full((3,len(offsets),len(maskOnset)),np.nan)
    testScore = trainScore.copy()
    for i,lbl in enumerate(dataType):
        for j,offset in enumerate(offsets):
            if offset is None:
                if lbl=='psth':
                    Xtrain,Xtest = X['train'],X['test']
                else:
                    break
            else:
                ind = offset if lbl=='bin' else slice(0,offset+1)
                Xdata = []
                for trialSet in ('train','test'):
                    d = np.reshape(X[trialSet],(len(y),len(units),-1))[:,:,ind]
                    if lbl=='count':
                        d = d.sum(axis=2)*binSize
                    Xdata.append(np.reshape(d,(len(y),-1)))
                Xtrain,Xtest = Xdata
            decoder.fit(Xtrain,y)
            
            for k,mo in enumerate(maskOnset):
                ind = m==mo
                trainScore[i][j][k] = decoder.score(Xtrain[ind],y[ind])
                testScore[i][j][k] = decoder.score(Xtest[ind],y[ind])
            if lbl=='psth' and offset==offsets[-1]:
                coef = np.mean(np.reshape(np.absolute(decoder.coef_),(len(units),-1)),axis=0)
    
    return trainScore,testScore,coef


maskOnset = [0,2,3,4,6]
analysisWindow = (t>0) & (t<0.2)
unitInd = np.where(respUnits & ~fs)[0]
unitInd = np.where(respUnits)[0]
nUnits = len(unitInd)

unitSessionInd = np.array([i for i,obj in enumerate(exps) for _ in enumerate(obj.goodUnits)])[unitInd]
unitSampleSize = {}
unitSampleSize['sessionCorr'] = [np.sum(unitSessionInd==i) for i in range(len(exps))]
unitSampleSize['sessionRand'] = unitSampleSize['sessionCorr']
unitSampleSize['pooled'] = [1,5,10,20,40,nUnits]
decoderOffset = np.arange(analysisWindow.sum())
trainTestIters = 100
trialsPerIter = 100

decodingVariable = ('side','choice')
unitSource = list(unitSampleSize.keys())
dataType = ('psth','bin','count')
trainScore = {var: {src: np.full((trainTestIters,len(unitSampleSize[src]),len(dataType),len(decoderOffset),len(maskOnset)),np.nan) for src in unitSource} for var in decodingVariable}
testScore = copy.deepcopy(trainScore)
decoderCoef = {var: {src: np.full((trainTestIters,len(unitSampleSize[src]),analysisWindow.sum()),np.nan) for src in unitSource} for var in decodingVariable}
for var,usource,dPsth,maskOnset in zip(decodingVariable,(unitSource,('pooled',)),(trialPsth,behavTrialPsth),([0,2,3,4,6],[2])):
    if var=='side':
        continue
    for iterInd in range(trainTestIters):
        # assign trials to training and testing sets
        varLabels = hemiLabels if var=='side' else ('correct','incorrect')
        trainInd = {vlbl: {mo: [] for mo in maskOnset} for vlbl in varLabels}
        testInd = copy.deepcopy(trainInd)
        for vlbl in varLabels:
            for mo in maskOnset:
                stim = 'targetOnly' if mo==0 else 'mask'
                for session in range(len(exps)):
                    units = unitInd[unitSessionInd==session]
                    p = dPsth[stim][vlbl]['all'] if var=='side' else dPsth[stim]['contra'][vlbl]
                    n = len(p[mo][units[0]])
                    trials = np.arange(n)
                    train = np.random.choice(trials,n//2,replace=False)
                    test = np.setdiff1d(trials,train)
                    for u in units:
                        trainInd[vlbl][mo].append(train)
                        testInd[vlbl][mo].append(test)
        
        for src in usource:
            for s,sampleSize in enumerate(unitSampleSize[src]):
                if src=='pooled':
                    if var=='choice':
                        unitSamples = [[u for u in range(nUnits) if all([len(ind[vlbl][mo][u])>0 for ind in (trainInd,testInd) for vlbl in varLabels for mo in maskOnset])]]
                    elif sampleSize==nUnits:
                        unitSamples = [np.arange(nUnits)]
                    elif sampleSize==1:
                        unitSamples = [[u] for u in range(nUnits)]
                    else:
                        # >99% chance each neuron is chosen at least once
                        nsamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                        unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nsamples)]
                    numTrials = trialsPerIter
                    randomizeTrials = True
                else:
                    unitSamples = [np.where(unitSessionInd==s)[0]]
                    numTrials = 10000
                    for trialSet,trialInd in zip(('train','test'),(trainInd,testInd)):
                        for hemi,rd in zip(hemiLabels,rewardDir):
                            for mo in maskOnset:
                                numTrials = min(numTrials,len(trialInd[vlbl][mo][unitSamples[0][0]]))
                    randomizeTrials = False if src=='sessionCorr' else True
                    
                trainSamples = []
                testSamples = []
                coefSamples = []
                for units in unitSamples:
                    offsets = [None] if src=='pooled' and sampleSize<max(unitSampleSize[src]) else decoderOffset
                    train,test,coef = getDecoderResult(var,varLabels,units,dPsth,trainInd,testInd,numTrials,randomizeTrials,analysisWindow,dataType,offsets,maskOnset)
                    trainSamples.append(train)
                    testSamples.append(test)
                    coefSamples.append(coef)
                trainScore[var][src][iterInd,s] = np.mean(trainSamples,axis=0)
                testScore[var][src][iterInd,s] = np.mean(testSamples,axis=0)
                decoderCoef[var][src][iterInd,s] = np.mean(coefSamples,axis=0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for score,clr,trialSet in zip((trainScore,testScore),('0.5','k'),('train','test')):
#    for src,mrk,mfc in zip(unitSource,'soo',('none','none',clr)):
    for src,mrk,mfc in zip(unitSource[1:],'oo',('none',clr)):
        overallScore = score[src][:,:,0,-1].mean(axis=-1)
        mean = overallScore.mean(axis=0)
        sem = overallScore.std(axis=0)/(trainTestIters**0.5)
#        if src=='sessionCorr':
#            srcLabel = 'session'
#        elif src=='sessionRand':
#            srcLabel = 'session, shuffled'
#        else:
#            srcLabel = src
        srcLabel = 'individual' if src=='sessionRand' else 'pooled'
        lbl = trialSet+', '+srcLabel
        ax.plot(unitSampleSize[src],mean,mrk,mec=clr,mfc=mfc,mew=2,ms=12,label=lbl)  
#        for x,m,s in zip(unitSampleSize[src],mean,sem):
#            ax.plot([x,x],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_ylim([0.5,1.02])
ax.set_xlabel('Number of Units',fontsize=16)
ax.set_ylabel('Decoder Accuracy',fontsize=16)
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x,y =  [testScore[src][:,:,0,-1].mean(axis=(0,-1)) for src in ('sessionCorr','sessionRand')]
ax.plot([0,1],[0,1],'--',color='0.8')
ax.plot(x,y,'o',mec='k',mfc='none',mew=2,ms=12)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0.5,1])
ax.set_ylim([0.5,1])
ax.set_aspect('equal')
ax.set_xlabel('Decoder Accuracy (Correlations Preserved)',fontsize=16)
ax.set_ylabel('Decoder Accuracy (Shuffled Trials)',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.concatenate(([8],maskOnset[1:]))/frameRate*1000
xticklabels = ['target only']+[str(int(round(x))) for x in xticks[1:]]
for score,clr,lbl in zip((trainScore,testScore),('0.5','k'),('train','test')):
    d = score['pooled'][:,-1,0,-1]
    mean = d.mean(axis=0)
    sem = d.std(axis=0)/(trainTestIters**0.5)   
    ax.plot(xticks,mean,'o',mec=clr,mfc=clr,label=lbl)  
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylim([0.5,1.02])
ax.set_xlabel('Mask Onset Relative to Target Onset (ms)',fontsize=12)
ax.set_ylabel('Decoder Accuracy',fontsize=12)
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d = decoderCoef['pooled'][:,-1]
m = d.mean(axis=0)
s = d.std(axis=0)/(trainTestIters**0.5)
ax.plot(t[analysisWindow]*1000,m,color='k')
ax.fill_between(t[analysisWindow]*1000,m+s,m-s,color='k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,200])
ax.set_xlabel('Time Relative to Target Onset (ms)',fontsize=12)
ax.set_ylabel('Decoder Weighting',fontsize=12)
plt.tight_layout()

clrs = np.zeros((len(maskOnset),3))
clrs[1:] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = ['target only']+[lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]
for i,xlbl in enumerate(('End of Decoding Window','Time','End of Spike Integration Window')): 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,200],[0.5,0.5],'k--')
    for j,(mo,clr,lbl) in enumerate(zip(maskOnset,clrs,lbls)):
        d = testScore['pooled'][:,-1,i,:,j]
        m = d.mean(axis=0)
        s = d.std(axis=0)/(trainTestIters**0.5)
        ax.plot(t[analysisWindow]*1000,m,color=clr,label=lbl)
        ax.fill_between(t[analysisWindow]*1000,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([0,200])
    ax.set_ylim([0.4,1])
    ax.set_xlabel(xlbl+' (ms)',fontsize=16)
    ax.set_ylabel('Target Side Decoding Accuracy',fontsize=16)
#    ax.legend(title='mask onset',fontsize=12)
    plt.tight_layout()
    

# choice decoding
clrs = np.zeros((len(maskOnset),3))
clrs[1:] = plt.cm.plasma(np.linspace(0,0.85,len(maskOnset)-1))[::-1,:3]
lbls = ['target only']+[lbl+' ms' for lbl in xticklabels[1:len(maskOnset)]]
for i,xlbl in enumerate(('End of Decoding Window','Time','End of Spike Integration Window')): 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,200],[0.5,0.5],'k--')
    for j,(mo,clr,lbl) in enumerate(zip(maskOnset,clrs,lbls)):
        d = testScore['choice']['pooled'][:,-1,i,:,j]
        m = d.mean(axis=0)
        s = d.std(axis=0)/(trainTestIters**0.5)
        ax.plot(t[analysisWindow]*1000,m,color=clr,label=lbl)
        ax.fill_between(t[analysisWindow]*1000,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([0,200])
    ax.set_ylim([0.4,1])
    ax.set_xlabel(xlbl+' (ms)',fontsize=16)
    ax.set_ylabel('Target Side Decoding Accuracy',fontsize=16)
    ax.legend(title='mask onset',fontsize=12)
    plt.tight_layout()


# plot response to optogenetic stimuluation during catch trials
optoPsthExample = []
i = 0
for obj in exps:
    validTrials = ~obj.longFrameTrials
    optoOnsetToPlot = np.nanmin(obj.optoOnset)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        for onset in [optoOnsetToPlot]:
            trials = validTrials & (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            p,t = getPsth(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
        t -= preTime
        optoPsthExample.append(p)
        i += 1
optoPsthExample = np.array(optoPsthExample)

optoPsth = []
baseRate = np.full(len(unitPos),np.nan)
stdBaseRate = baseRate.copy()
transientOptoResp = baseRate.copy()
sustainedOptoResp = baseRate.copy()
sustainedOptoRate = baseRate.copy()
i = 0
for obj in exps:
    validTrials = ~obj.longFrameTrials
    for u in obj.goodUnits:
        p = []
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        for onset in np.unique(obj.optoOnset[~np.isnan(obj.optoOnset)]):
            trials = validTrials & (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            s,t = getPsth(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
            p.append(s)
        t -= preTime
        p = np.mean(np.concatenate(p),axis=0)
        optoPsth.append(p)
        baseRate[i] = p[t<0].mean()
        stdBaseRate[i] = p[t<0].std()
        transientOptoResp[i] = p[(t>0) & (t<0.1)].max()-baseRate[i]
        sustainedOptoRate[i] = p[(t>0.1) & (t<0.5)].mean()
        sustainedOptoResp[i] = sustainedOptoRate[i]-baseRate[i]
        i += 1
optoPsth = np.array(optoPsth)

peakThresh = 5*stdBaseRate
sustainedThresh = stdBaseRate
excit = sustainedOptoResp>sustainedThresh
transient = (~excit) & (transientOptoResp>peakThresh)
inhib = (~transient) & (sustainedOptoResp<sustainedThresh)
noResp = ~(excit | inhib | transient)

for k,p in enumerate((optoPsthExample,optoPsth)):
    fig = plt.figure(figsize=(6,8))
    for i,(units,lbl) in enumerate(zip((excit,transient,inhib,inhib),('Excited','Transiently Excited','Inhibited',''))):
        ax = fig.add_subplot(4,1,i+1)
        m = p[units].mean(axis=0)
        s = p[units].std(axis=0)/(units.sum()**0.5)
        ax.plot(t,m,'k')
        ax.fill_between(t,m+s,m-s,color='k',alpha=0.25)
        ylim = plt.get(ax,'ylim')
        if i==2:
            for x in (-0.01,0.05):
                ax.plot([x,x],[0,100],'--',color='0.5',zorder=1)
        if k==0:
            poly = np.array([(0,0),(trialTime-optoOnsetToPlot/obj.frameRate+0.1,0),(trialTime-optoOnsetToPlot/obj.frameRate,ylim[1]),(0,ylim[1])])
            ax.add_patch(matplotlib.patches.Polygon(poly,fc='c',ec='none',alpha=0.1))
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        if i<3:
            n = str(np.sum(units & fs))+' FS, '+str(np.sum(units & ~fs))+' RS'
            ax.text(0.5,1,lbl+' (n = '+n+')',transform=ax.transAxes,color='k',ha='center',va='top',fontsize=12)
            ax.set_xlim([-0.2,1])
        else:
            ax.set_xlabel('Time From Optogenetic Light Onset (s)',fontsize=12)
            ax.set_xlim([-0.01,0.05])
        ax.set_ylim([0,ylim[1]])
        ax.set_ylabel('Spikes/s',fontsize=12)
    plt.tight_layout()

fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(4,2)
for j,(xdata,xlbl) in enumerate(zip((peakToTrough,unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
    for i,(y,ylbl) in enumerate(zip((baseRate,transientOptoResp,sustainedOptoResp,sustainedOptoRate),('Baseline rate','Transient opto response','Sustained opto response','Sustained opto rate'))):
        ax = fig.add_subplot(gs[i,j])
        xmin = xdata.min()
        xmax = xdata.max()
        xrng = xmax-xmin
        xlim = [xmin-0.05*xrng,xmax+0.05*xrng]
        if i in (1,2):
            ax.plot(xlim,[0,0],'--',color='0.6')
        for ind,clr in zip((fs,~fs),'mg'):
            ax.plot(xdata[ind],y[ind],'o',mec=clr,mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(xlim)
        if i==3:
            ax.set_xlabel(xlbl,fontsize=12)
        if j==0:
            ax.set_ylabel(ylbl,fontsize=12)
        if i==0 and j==0:
            for x,lbl,clr in zip((0.25,0.75),('FS','RS'),'mg'):
                ax.text(x,1.05,lbl,transform=ax.transAxes,color=clr,fontsize=12,ha='center',va='bottom')
plt.tight_layout()



# plot response to visual stimuli with opto
optoStimLabels = ['targetOnly','mask','maskOnly','catch']
optoOnset = list(np.unique(exps[0].optoOnset[~np.isnan(exps[0].optoOnset)]))+[np.nan]
optoOnsetPsth = {stim: {hemi: {onset: [] for onset in optoOnset} for hemi in hemiLabels} for stim in optoStimLabels}
for obj in exps:
    validTrials = ~obj.longFrameTrials
    ephysHemi = hemiLabels[::-1] if hasattr(obj,'hemi') and obj.hemi=='right' else hemiLabels
    for stim in stimLabels:
        for hemi,rd in zip(ephysHemi,rewardDir):
            stimTrials = np.in1d(obj.trialType,(stim,stim+'Opto'))
            if stim in ('targetOnly','mask'):
                stimTrials = stimTrials & (obj.rewardDir==rd)
            for mo in np.unique(obj.maskOnset[stimTrials]):
                moTrials = obj.maskOnset==mo
                for onset in optoOnset:
                    optoTrials = np.isnan(obj.optoOnset) if np.isnan(onset) else obj.optoOnset==onset
                    trials = validTrials & stimTrials & moTrials & optoTrials
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for u in obj.goodUnits:
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPsth(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                        optoOnsetPsth[stim][hemi][onset].append(p)                      
t -= preTime

analysisWindow = (t>0.04) & (t<0.12)
          
units = ~(transient | excit) & respUnits
                
optoOnsetTicks = list(1000*(np.array(optoOnset[:-1])-exps[0].frameDisplayLag)/frameRate) + [100]
optoOnsetLabels = [str(int(round(onset))) for onset in optoOnsetTicks[:-1]] + ['no\nopto']

fig = plt.figure(figsize=(6,8))
gs = matplotlib.gridspec.GridSpec(3,2)
axs = []
cmap = np.zeros((len(optoOnset),3))
cint = 1/(len(optoOnset)-1)
cmap[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
cmap[:-1,2] = 1
for i,(stim,stimLbl) in enumerate(zip(optoStimLabels,('target','target + mask','mask only','no visual stimulus'))):
    for j,hemi in enumerate(hemiLabels):
        if stim in ('maskOnly','catch'):
            loc = gs[i,0] if stim=='maskOnly' else gs[i-1,1]
        else:
            loc = gs[i,j]
        ax = fig.add_subplot(loc)
        axs.append(ax)
        for onset,clr,lbl in zip(optoOnset,cmap,optoOnsetLabels):
            p = np.array(optoOnsetPsth[stim][hemi][onset])[units]
            m = np.mean(p,axis=0)
            s = np.std(p,axis=0)/(len(p)**0.5)
            lbl = lbl.replace('\n',' ') if np.isnan(onset) else lbl+' ms'
            ax.plot(t*1000,m,color=clr,label=lbl)
            ax.fill_between(t*1000,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-50,201,50))
        ax.set_xlim([-25,175])
        if i==2 and j==0:
            ax.set_xlabel('Time From Visual Stimulus Onset (ms)',fontsize=12)
        if i==1 and j==0:
            ax.set_ylabel('Spikes/s',fontsize=12)
        if i==0 and j==1:
            ax.legend(loc='upper right',title='opto onset',fontsize=8)
        title = hemi+' '+stimLbl if stim in ('targetOnly','mask') else stimLbl
        ax.text(0.05,1,title,transform=ax.transAxes,color='k',ha='left',va='top',fontsize=12)
        if stim in ('maskOnly','catch'):
            break
ymin = min([plt.get(ax,'ylim')[0] for ax in axs]+[0])
ymax = max(plt.get(ax,'ylim')[1] for ax in axs)
for ax in axs:
    ax.set_ylim([ymin,ymax])
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for stim,clr,lbl in zip(optoStimLabels[:3],'kbg',('target only','target + mask','mask only')):
    respMean = []
    respSem = []
    for onset in optoOnset:
        p = np.array(optoOnsetPsth[stim]['contra'][onset])[units]
        c = np.array(optoOnsetPsth['catch']['contra'][onset])[units]
        r = (p-c)[:,analysisWindow].sum(axis=1)*binSize
        respMean.append(np.mean(r))
        respSem.append(np.std(r)/(len(r)**0.5))
    ax.plot(optoOnsetTicks,respMean,'o',color=clr,label=lbl)
    for x,m,s in zip(optoOnsetTicks,respMean,respSem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(optoOnsetTicks)
ax.set_xticklabels(optoOnsetLabels)
ax.set_xlim([8,108])
ax.set_xlabel('Optogenetic light onset relative to target onset (ms)',fontsize=12)
ax.set_ylabel('Stimulus evoked spikes per neuron',fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for stim,clr in zip(('targetOnly','mask'),'kb'):
    mean = []
    sem = []
    for onset in optoOnset:
        d = []
        ipsi,contra = [np.array(optoOnsetPsth[stim][hemi][onset])[units][:,analysisWindow].sum(axis=1)*binSize for hemi in ('ipsi','contra')]
        for i,c in zip(ipsi,contra):
            if c==0:
                d.append(0.5)
            else:
                d.append(c/(c+i))
#        d = contra-ipsi
        mean.append(np.mean(d))
        sem.append(np.std(d)/(len(d)**0.5))
    if stim=='targetOnly':
        firstValid = 2
    elif stim=='mask':
        firstValid = 2
    ax.plot(optoOnsetTicks[firstValid:],mean[firstValid:],'o',color=clr)
    for x,m,s in zip(optoOnsetTicks[firstValid:],mean[firstValid:],sem[firstValid:]):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(optoOnsetTicks)
ax.set_xticklabels(optoOnsetLabels)
ax.set_yticks(np.arange(0.4,1,0.1))
ax.set_xlim([8,108])
ax.set_ylim([0.4,0.8])
ax.set_xlabel('Optogenetic light onset relative to target onset (ms)',fontsize=12)
ax.set_ylabel('Fraction of spikes contralateral to target',fontsize=12)
plt.tight_layout()



# LFP
import scipy.io
f = fileIO.getFile('load channel map',fileType='*.mat')
channelMapData = scipy.io.loadmat(f)
channelMap = channelMapData['chanMap0ind'].flatten()
ycoords = channelMapData['ycoords']

unitSessionInd = np.array([i for i,obj in enumerate(exps) for _ in enumerate(obj.goodUnits)])

for i,obj in enumerate(exps):
    probeData,analogInData = loadDatData(obj.datFilePath)
    
    preSamples = int(0*obj.sampleRate)
    postSamples = int(0.2*obj.sampleRate)
    resp = np.zeros((128,preSamples+postSamples))
    trials = obj.trialType=='maskOnly'
    startSamples = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]
    for n,s in enumerate(startSamples):
        resp += probeData[channelMap[:128],s-preSamples:s+postSamples]
    resp /= n+1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(0,preSamples+postSamples,int(0.05*obj.sampleRate))
    xticklabels = (xticks-preSamples)/obj.sampleRate
    ax.imshow(resp,aspect='auto')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    print(np.unique([obj.units[u]['peakChan'] for u in np.array(obj.goodUnits)[respUnits[unitSessionInd==i]]]))



# rf mapping
rfExps = []
while True:
    f = fileIO.getFile('choose rf data file',fileType='*.hdf5')
    if f=='':
        break
    else:
        obj = MaskTaskData()
        obj.loadEphysData(led=False)
        obj.loadKilosortData(os.path.join(os.path.dirname(obj.datFilePath),'kilosort'))
        obj.loadRFData(f)
        rfExps.append(obj)


for obj in rfExps:
    azi,ele = [np.unique(p) for p in obj.rfStimPos.T]
    ori = obj.rfOris
    contrast = np.unique(obj.rfStimContrast)
    dur = np.unique(obj.rfStimFrames)
    binSize = 1/obj.frameRate
    preTime = 0.1
    postTime = 0.6
    nbins = np.arange(0,preTime+postTime+binSize,binSize).size-1
    units =obj.sortedUnits
    rfMap = np.zeros((units.size,nbins,ele.size,azi.size,ori.shape[0],contrast.size,dur.size))
    for i,y in enumerate(ele):
        eleTrials = obj.rfStimPos[:,1]==y
        for j,x in enumerate(azi):
            aziTrials = obj.rfStimPos[:,0]==x
            for k,o in enumerate(ori):
                oriTrials = obj.rfStimOri[:,0]==o[0]
                if np.isnan(o[1]):
                    oriTrials = oriTrials & np.isnan(obj.rfStimOri[:,1])
                else:
                    oriTrials = oriTrials & (obj.rfStimOri[:,1]==o[1])
                for l,c in enumerate(contrast):
                    contrastTrials = obj.rfStimContrast==c
                    for m,d in enumerate(dur):
                        trials = eleTrials & aziTrials & oriTrials & contrastTrials & (obj.rfStimFrames==d)
                        startTimes = obj.frameSamples[obj.rfStimStart[trials]]/obj.sampleRate
                        for n,u in enumerate(units):
                            spikeTimes = obj.units[u]['samples']/obj.sampleRate
                            p,t = getPsth(spikeTimes,startTimes-preTime,preTime+postTime,binSize=binSize,avg=True)
                            t -= preTime
    #                        p -= p[t<0].mean()
                            rfMap[n,:,i,j,k,l,m] = p
    
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(ele.size,azi.size)
    m = rfMap.mean(axis=(0,4,5,6))
    ymax = m.max()
    for i,y in enumerate(ele):
        for j,x in enumerate(azi):
            ax = fig.add_subplot(gs[ele.size-1-i,j])
            ax.plot(t,m[:,i,j],'k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            if i==0 and j==1:
                ax.set_xlabel('Time from stim onset (s)')
            else:
                ax.set_xticklabels([])
            if i==1 and j==0:
                ax.set_ylabel('Spikes/s')
            else:
                ax.set_yticklabels([])
            ax.set_xticks([0,0.5])
            ax.set_yticks([0,10,20,30])
            ax.set_ylim([0,1.02*ymax])
    plt.tight_layout()



# spike rate autocorrelation
corrWidth = 50
corr = np.zeros((sum([len(obj.goodUnits) for obj in exps]),corrWidth*2+1))
uind = 0
for obj in exps:
    starts = np.round(1000*np.concatenate(([0],obj.frameSamples[obj.stimStart]/obj.sampleRate+obj.responseWindowFrames/obj.frameRate))).astype(int)
    stops = np.round(np.concatenate((obj.frameSamples[obj.stimStart]/obj.sampleRate,[obj.frameSamples[-1]/obj.sampleRate]))*1000-corrWidth).astype(int)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate*1000
        c = np.zeros(corrWidth*2+1)
        n = 0
        for i,j in zip(starts,stops):
            spikes = spikeTimes[(spikeTimes>i) & (spikeTimes<j)]
            for s in spikes:
                bins = np.arange(s-corrWidth-0.5,s+corrWidth+1)
                c += np.histogram(spikes,bins)[0]
                n += 1
        c /= n
        corr[uind] = c
        uind += 1
        print(str(uind)+'/'+str(corr.shape[0]))
        

expfunc = lambda x,a,tau,c: a*np.exp(-x/tau)+c

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
t = np.arange(corrWidth*2+1)-corrWidth
x0 = 3
fitInd = slice(corrWidth+x0,None)
fitx = t[fitInd]-x0
#tau = []
#for c in corr:
#    if not np.all(np.isnan(c)):
#        ax.plot(t,c,color='0.5')
#        fitParams = scipy.optimize.curve_fit(expfunc,fitx,c[fitInd])[0]
#        tau.append(fitParams[1])
#        ax.plot(t[fitInd],expfunc(fitx,*fitParams),color=[1,0.5,0.5])
m = np.nanmean(corr,axis=0)
ax.plot(t,m,'k')
fitParams = scipy.optimize.curve_fit(expfunc,fitx,m[fitInd])[0]
tauMean = fitParams[1]
#ax.plot(t[fitInd],expfunc(fitx,*fitParams),'r')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
#ax.set_ylim([0,1.05*np.nanmax(corr[:,t!=0])])
ax.set_xlim([0,corrWidth])
ax.set_ylim([0,0.04])
ax.set_yticks([0,0.01,0.02,0.03,0.04])
ax.set_xlabel('Lag (ms)')
ax.set_ylabel('Autocorrelation')
plt.tight_layout()

