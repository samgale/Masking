# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import os
import h5py
import numpy as np
import pandas as pd
import scipy.signal
import scipy.ndimage
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from numba import njit
import fileIO


@njit
def findSignalEdges(signal,edgeType,thresh,refractory):
    """
    signal: typically a large memmap array (loop through values rather than load all into memory)
    edgeType: 'rising' or 'falling'
    thresh: difference between current and previous value
    refractory: samples after detected edge to ignore
    """
    edges = []
    lastVal = signal[0]
    lastEdge = -refractory
    for i in range(1,signal.size):
        val = signal[i]
        if i-lastEdge>refractory and ((edgeType=='rising' and val-lastVal>thresh) or (edgeType=='falling' and val-lastVal<thresh)):
            edges.append(i)
            lastEdge = i
        lastVal = val
    return edges


def getPSTH(spikes,startTimes,windowDur,binSize=0.01,avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts, bins[:-1]+binSize/2


def getSDF(spikes,startTimes,windowDur,sampInt=0.001,filt='exponential',filtWidth=0.005,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
        if filt in ('exp','exponential'):
            filtPts = int(5*filtWidth/sampInt)
            expFilt = np.zeros(filtPts*2)
            expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=filtWidth/sampInt,sym=False)
            expFilt /= expFilt.sum()
            sdf = scipy.ndimage.filters.convolve1d(counts,expFilt,axis=1)
        else:
            sdf = scipy.ndimage.filters.gaussian_filter1d(counts,filtWidth/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf,t[:-1]



# get analog sync data acquired with NidaqRecorder
syncPath = fileIO.getFile('Select sync file',fileType='*.hdf5')
syncFile = h5py.File(syncPath,'r')
syncData = syncFile['AnalogInput']
syncSampleRate = syncData.attrs.get('sampleRate')
channelNames = syncData.attrs.get('channelNames')
vsync = syncData[:,channelNames=='vsync'][:,0]
photodiode = syncData[:,channelNames=='photodiode'][:,0]
led = syncData[:,channelNames=='led'][:,0]
syncTime = np.arange(1/syncSampleRate,(syncData.shape[0]+1)/syncSampleRate,1/syncSampleRate)

syncFile.close()

frameSamples = np.array(findSignalEdges(vsync,edgeType='falling',thresh=-0.5,refractory=2))

behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')

psychopyFrameIntervals = behavData['frameIntervals'][:]
frameRate = round(1/np.median(psychopyFrameIntervals))

assert(frameSamples.size==psychopyFrameIntervals.size+1)

ntrials = behavData['trialEndFrame'].size
stimStart = behavData['trialStimStartFrame'][:ntrials]
trialOpenLoopFrames = behavData['trialOpenLoopFrames'][:ntrials]
assert(np.unique(trialOpenLoopFrames).size==1)
openLoopFrames = trialOpenLoopFrames[0]
responseWindowFrames = behavData['maxResponseWaitFrames'][()]
optoOnset = behavData['trialOptoOnset'][:ntrials]
targetFrames = behavData['trialTargetFrames'][:ntrials]
maskFrames = behavData['trialMaskFrames'][:ntrials]
maskOnset = behavData['trialMaskOnset'][:ntrials]

behavData.close()

optoOnsetToPlot = 0
opto = optoOnset==optoOnsetToPlot

stimDur = []
for st in stimStart:
    stimDur.append(psychopyFrameIntervals[st+1:st+3].sum())
stimDur = np.array(stimDur)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
samples = np.arange(frameSamples[0]-100,frameSamples[0]+201)
t = (samples-frameSamples[0])/syncSampleRate
ax.plot(t,vsync[samples],color='k',label='vsync')
ax.plot(t,photodiode[samples],color='0.5',label='photodiode')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from first frame (s)')
ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ind = frameSamples[stimStart[np.where(opto)[0][0]]]
samples = np.arange(ind-1500,ind+3001)
t = (samples-ind)/syncSampleRate
ax.plot(t,vsync[samples],color='k',label='vsync')
ax.plot(t,photodiode[samples],color='0.5',label='photodiode')
ax.plot(t,led[samples],color='b',label='led')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.005,0.01])
ax.set_xlabel('Time from trial start (s)')
ax.legend()
plt.tight_layout()




# get probe data
probeDataDir = fileIO.getDir()

sampleRate = 30000
totalChannels = 136
probeChannels = 128      

rawData = np.memmap(os.path.join(probeDataDir,'continuous.dat'),dtype='int16',mode='r')    
rawData = np.reshape(rawData,(int(rawData.size/totalChannels),-1)).T
totalSamples = rawData.shape[1]

analogInData = {name: rawData[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                          'photodiode',
                                                                          'rotaryEncoder',
                                                                          'cam1Exposure',
                                                                          'cam2Exposure',
                                                                          'led1',
                                                                          'led2'))}

kilosortData = {key: np.load(os.path.join(probeDataDir,'kilosort',key+'.npy')) for key in ('spike_clusters',
                                                                                           'spike_times',
                                                                                           'templates',
                                                                                           'spike_templates',
                                                                                           'channel_positions',
                                                                                           'amplitudes')}

clusterIDs = pd.read_csv(os.path.join(probeDataDir,'kilosort','cluster_KSLabel.tsv'),sep='\t')

unitIDs = np.unique(kilosortData['spike_clusters'])

units = {}
for u in unitIDs:
    units[u] = {}
    units[u]['label'] = clusterIDs[clusterIDs['cluster_id']==u]['KSLabel'].tolist()[0]
    
    uind = np.where(kilosortData['spike_clusters']==u)[0]
    
    units[u]['samples'] = kilosortData['spike_times'][uind].flatten()
    
    #choose 1000 spikes with replacement, then average their templates together
    chosen_spikes = np.random.choice(uind,1000)
    chosen_templates = kilosortData['spike_templates'][chosen_spikes].flatten()
    units[u]['template'] = np.mean(kilosortData['templates'][chosen_templates],axis=0)
    
    peakChan = np.unravel_index(np.argmin(units[u]['template']),units[u]['template'].shape)[1]
    units[u]['peakChan'] = peakChan
    units[u]['position'] = kilosortData['channel_positions'][peakChan]
    units[u]['amplitudes'] = kilosortData['amplitudes'][uind]
    
    template = units[u]['template'][:,peakChan]
    if any(np.isnan(template)):
        units[u]['peakToTrough'] = np.nan
    else:
        peakInd = np.argmin(template)
        units[u]['peakToTrough'] = np.argmax(template[peakInd:])/(sampleRate/1000)
    
    #check if this unit is noise
    tempNorm = template/np.max(np.absolute(template))
    units[u]['normTempIntegral'] = tempNorm.sum()
    if abs(tempNorm.sum())>4:
        units[u]['label'] = 'noise'
        
goodUnits = np.array([u for u in units if units[u]['label']=='good'])
epochs = 4
epochSamples = totalSamples/epochs
hasSpikes = np.ones(len(goodUnits),dtype=bool)
for i in range(epochs):
    hasSpikes = hasSpikes & (np.array([np.sum((units[u]['samples']>=i*epochSamples) & (units[u]['samples']<i*epochSamples+epochSamples))/epochSamples*sampleRate for u in goodUnits]) > 0.1)
goodUnits = goodUnits[hasSpikes]
goodUnits = goodUnits[np.argsort([units[u]['peakChan'] for u in goodUnits])]

peakToTrough = np.array([units[u]['peakToTrough'] for u in goodUnits])
fs = peakToTrough<=0.5
unitPos = np.array([units[u]['position'][1]/1000 for u in goodUnits])


# get behavior data    
behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')
ntrials = behavData['trialEndFrame'].size
stimStart = behavData['trialStimStartFrame'][:ntrials]
trialOpenLoopFrames = behavData['trialOpenLoopFrames'][:ntrials]
assert(np.unique(trialOpenLoopFrames).size==1)
openLoopFrames = trialOpenLoopFrames[0]
responseWindowFrames = behavData['maxResponseWaitFrames'][()]
trialType = behavData['trialType'][:ntrials]
targetFrames = behavData['trialTargetFrames'][:ntrials]
maskFrames = behavData['trialMaskFrames'][:ntrials]
maskOnset = behavData['trialMaskOnset'][:ntrials]
optoOnset = behavData['trialOptoOnset'][:ntrials]
rewardDir = behavData['trialRewardDir'][:ntrials]


# get frame times and compare with psychopy frame intervals
frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))

psychopyFrameIntervals = behavData['frameIntervals'][:]
frameRate = round(1/np.median(psychopyFrameIntervals))

assert(frameSamples.size==psychopyFrameIntervals.size+1)

# check frame display lag
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
samples = np.arange(frameSamples[0]-1500,frameSamples[0]+3001)
t = (samples-frameSamples[0])/sampleRate
ax.plot(t,analogInData['vsync'][samples],color='k',label='vsync')
ax.plot(t,analogInData['photodiode'][samples],color='0.5',label='photodiode')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from first frame (s)')
ax.legend()
plt.tight_layout()

frameDisplayLag = 2


# plot response to optogenetic stimuluation during catch trials
psth = []
peakBaseRate = np.full(goodUnits.size,np.nan)
meanBaseRate = peakBaseRate.copy()
transientOptoResp = peakBaseRate.copy()
sustainedOptoResp = peakBaseRate.copy()
sustainedOptoRate = peakBaseRate.copy()
preTime = 0.5
postTime = 0.5
trialTime = (openLoopFrames+responseWindowFrames)/frameRate
windowDur = preTime+trialTime+postTime
binSize = 1/frameRate
optoOnsetToPlot = np.nanmin(optoOnset)
for i,u in enumerate(goodUnits):
    spikeTimes = units[u]['samples']/sampleRate
    p = []
    for onset in [optoOnsetToPlot]: #np.unique(optoOnset[~np.isnan(optoOnset)]):
        trials = (trialType=='catchOpto') & (optoOnset==onset)
        startTimes = frameSamples[stimStart[trials]+int(onset)]/sampleRate-preTime
        s,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
        p.append(s)
    p = np.mean(np.concatenate(p),axis=0)
    psth.append(p)
    peakBaseRate[i] = p[t<preTime].max()
    meanBaseRate[i] = p[t<preTime].mean()
    transientOptoResp[i] = p[(t>=preTime) & (t<preTime+0.25)].max()-peakBaseRate[i]
    sustainedOptoRate[i] = p[(t>preTime+trialTime-0.35) & (t<preTime+trialTime-0.25)].mean()
    sustainedOptoResp[i] = sustainedOptoRate[i]-meanBaseRate[i]
t -= preTime
psth = np.array(psth)

fig = plt.figure(figsize=(8,8))
excit = sustainedOptoResp>1
inhib = ((sustainedOptoResp<0) & (transientOptoResp<1))
transient = ~(excit | inhib) & (transientOptoResp>1)
noResp = ~(excit | inhib | transient)
gs = matplotlib.gridspec.GridSpec(4,2)
for i,j,clr,ind,lbl in zip((0,1,0,1,2,3),(0,0,1,1,1,1),'mgkkkk',(fs,~fs,excit,inhib,transient,noResp),('FS','RS','Excited','Inhibited','Transient','No Response')):
    ax = fig.add_subplot(gs[i,j])
    ax.plot(t,psth[ind].mean(axis=0),clr)
    ylim = plt.get(ax,'ylim')
    poly = np.array([(0,0),(trialTime-optoOnsetToPlot/frameRate+0.1,0),(trialTime,ylim[1]),(0,ylim[1])])
    ax.add_patch(matplotlib.patches.Polygon(poly,fc='c',ec='none',alpha=0.25))
    n = str(np.sum(ind)) if j==0 else str(np.sum(ind & fs))+' FS, '+str(np.sum(ind & ~fs))+' RS'
    ax.text(1,1,lbl+' (n = '+n+')',transform=ax.transAxes,color=clr,ha='right',va='bottom')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    if (i==1 and j==0) or i==2:
        ax.set_xlabel('Time from LED onset (s)')
    ax.set_ylabel('Spikes/s')
plt.tight_layout()

fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(4,2)
for j,(xdata,xlbl) in enumerate(zip((peakToTrough,unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
    for i,(y,ylbl) in enumerate(zip((meanBaseRate,transientOptoResp,sustainedOptoResp,sustainedOptoRate),('Baseline rate','Transient opto response','Sustained opto response','Sustained opto rate'))):
        ax = fig.add_subplot(gs[i,j])
        for ind,clr in zip((fs,~fs),'mg'):
            ax.plot(xdata[ind],y[ind],'o',mec=clr,mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i in (1,2):
            ax.plot(plt.get(ax,'xlim'),[0,0],'--',color='0.6',zorder=0)
        if i==3:
            ax.set_xlabel(xlbl)
        if j==0:
            ax.set_ylabel(ylbl)
        if i==0 and j==0:
            for x,lbl,clr in zip((0.25,0.75),('FS','RS'),'mg'):
                ax.text(x,1.05,lbl,transform=ax.transAxes,color=clr,fontsize=12,ha='center',va='bottom')
plt.tight_layout()


# plot response to visual stimuli without opto
stimLabels = ('targetOnly','maskOnly','mask')
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/frameRate
peakResp = {cellType: {stim:[] for stim in stimLabels} for cellType in ('FS','RS')}
timeToPeak = {cellType: {stim:[] for stim in stimLabels} for cellType in ('FS','RS')}
timeToFirstSpike = {cellType: {stim:[] for stim in stimLabels} for cellType in ('FS','RS')}
for ct,cellType in zip((fs,~fs),('FS','RS')):
    fig = plt.figure(figsize=(10,5))
    fig.text(0.5,0.99,cellType,ha='left',va='top',fontsize=12)
    axs = []
    for i,rd in enumerate((1,-1)):
        ax = fig.add_subplot(1,2,i+1)
        axs.append(ax)
        for stim,clr in zip(stimLabels,('k','0.5','r')):
            stimTrials = trialType==stim if stim=='maskOnly' else (trialType==stim) & (rewardDir==rd)
            mskOn = np.unique(maskOnset[stimTrials])
            if stim=='mask' and len(mskOn)>1:
                cmap = np.ones((len(mskOn),3))
                cint = 1/(len(mskOn)-1)
                cmap[:,1:] = np.arange(0,1.01-cint,cint)[:,None]
            else:
                cmap = [clr]
            for mo,c in zip(mskOn,cmap):
                moTrials = maskOnset==mo
                trials = stimTrials & (maskOnset==mo)
                startTimes = frameSamples[stimStart[trials]+frameDisplayLag]/sampleRate-preTime
                psth = []
                if rd==-1:
                    timeToFirstSpike[cellType][stim].append([])
                for u in goodUnits[ct]:
                    spikeTimes = units[u]['samples']/sampleRate
                    p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                    t -= preTime
                    p -= p[t<0].mean()
                    psth.append(p)
                    if rd==-1:
                        lat = []
                        for st in startTimes:
                            firstSpike = np.where((spikeTimes > st+0.03) & (spikeTimes < st+0.15))[0]
                            if len(firstSpike)>0:
                                lat.append(spikeTimes[firstSpike[0]]-st)
                            else:
                                lat.append(np.nan)
                        timeToFirstSpike[cellType][stim][-1].append(np.nanmedian(lat))
                psth = np.array(psth)
                if rd==-1:
                    analysisWindow = (t>0.03) & (t<0.15)
                    peakResp[cellType][stim].append(psth[:,analysisWindow].max(axis=1))
                    timeToPeak[cellType][stim].append(t[np.argmax(psth[:,analysisWindow],axis=1)+np.where(analysisWindow)[0][0]])
                m = np.mean(psth,axis=0)
                s = np.std(psth,axis=0)/(len(psth)**0.5)
                lbl = 'SOA '+str(round(1000*mo/frameRate,1))+' ms' if stim=='mask' else stim
                lbl += '; time to peak '+str(round(1000*t[np.argmax(m)],1)) + ' ms'
                ax.plot(t,m,color=c,label=lbl)
    #            ax.fill_between(t,m+s,m-s,color=c,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(0,0.151,0.05))
        ax.set_xlim([0,0.15])
        ax.set_xlabel('Time from stimulus onset (s)')
        if i==0:
            ax.set_ylabel('Response (spikes/s)')
        ax.legend(fontsize=6,loc='upper left',frameon=False)
        title = 'target left' if rd==1 else 'target right'
        ax.set_title(title)
    ymin = min([plt.get(ax,'ylim')[0] for ax in axs]+[0])
    ymax = max(plt.get(ax,'ylim')[1] for ax in axs)
    for ax in axs:
        ax.set_ylim([ymin,ymax])
    plt.tight_layout()
    

fig = plt.figure(figsize=(10,6))
gs = matplotlib.gridspec.GridSpec(2,2)
for j,(xdata,xlbl) in enumerate(zip((peakToTrough,unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
    for i,stim in enumerate(stimLabels[:2]):
        ax = fig.add_subplot(gs[i,j])
        for ct,cellType,clr in zip((fs,~fs),('FS','RS'),'mg'):
            ax.plot(xdata[ct],peakResp[cellType][stim][0],'o',mec=clr,mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i==1:
            ax.set_xlabel(xlbl)
        if j==0:
            ax.set_ylabel('Response to '+stim[:stim.find('Only')]+' (spikes/s)')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,200],[0,200],'--',color='0.5')
amax = 0
for cellType,clr in zip(('FS','RS'),'mg'):
    x,y = [peakResp[cellType][stim][0] for stim in stimLabels[:2]]
    amax = max(amax,x.max(),y.max())
    ax.plot(x,y,'o',mec=clr,mfc='none',label=cellType)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.02*amax,1.02*amax])
ax.set_ylim([-0.02*amax,1.02*amax])
ax.set_aspect('equal')
ax.set_xlabel('Response to target (spikes/s)')
ax.set_ylabel('Response to mask (spikes/s)')
ax.legend()
plt.tight_layout()


fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,2)
for j,cellType in enumerate(('FS','RS')):
    for i,ydata in enumerate((timeToPeak,timeToFirstSpike)):
        ax = fig.add_subplot(gs[i,j])
        for x,stim in enumerate(stimLabels):
            y = ydata[cellType][stim][0]
            ax.plot(x+np.zeros(len(y)),y,'o',mec='0.5',mfc='none')
            m = np.nanmean(y)
            s = np.nanstd(y)/(np.sum(~np.isnan(y))**0.5)
            ax.plot(x,m,'ko')
            ax.plot([x,x],[m-s,m+s],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(len(stimLabels)))
        ax.set_xticklabels(stimLabels)
        ax.set_ylim([0.025,0.155])



# plot response to visual stimuli with opto
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/frameRate
optOn = list(np.unique(optoOnset[~np.isnan(optoOnset)]))+[np.nan]
cmap = np.zeros((len(optOn),3))
cint = 1/(len(optOn)-1)
cmap[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
cmap[:-1,2] = 1
fig = plt.figure(figsize=(6,10))
axs = []
naxs = 2+np.sum(np.unique(maskOnset>0))
for stim in stimLabels:
    stimTrials = np.in1d(trialType,(stim,stim+'Opto'))
    if stim!='maskOnly':
        stimTrials = stimTrials & (rewardDir==-1)
    for mo in np.unique(maskOnset[stimTrials]):
        ax = fig.add_subplot(naxs,1,len(axs)+1)
        axs.append(ax)
        moTrials = maskOnset==mo
        for onset,clr in zip(optOn,cmap):
            trials = np.isnan(optoOnset) if np.isnan(onset) else optoOnset==onset
            trials = trials & stimTrials & moTrials
            startTimes = frameSamples[stimStart[trials]+frameDisplayLag]/sampleRate-preTime
            psth = []
            for u in goodUnits[~fs]:
                spikeTimes = units[u]['samples']/sampleRate
                p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                psth.append(p)
            t -= preTime
            m = np.mean(psth,axis=0)
            s = np.std(psth,axis=0)/(len(psth)**0.5)
            lbl = 'no opto' if np.isnan(onset) else str(int(round(1000*(onset-frameDisplayLag)/frameRate)))+' ms'
            ax.plot(t,m,color=clr,label=lbl)
#            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(-0.05,0.21,0.05))
        ax.set_xlim([-0.05,0.2])
        ax.set_ylabel('Spikes/s')
        title = 'SOA '+str(int(round(1000*mo/frameRate)))+' ms' if stim=='mask' else stim
        ax.set_title(title)
        if len(axs)==1:
            ax.legend(loc='upper right',title='opto onset')
        elif len(axs)==naxs:
            ax.set_xlabel('Time from stimulus onset (s)')
ymin = min([plt.get(ax,'ylim')[0] for ax in axs]+[0])
ymax = max(plt.get(ax,'ylim')[1] for ax in axs)
for ax in axs:
    ax.set_ylim([ymin,ymax])
plt.tight_layout()        




# bilateral opto
behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')

ntrials = behavData['trialEndFrame'].size
trialType = behavData['trialType'][:ntrials]
targetContrast = behavData['trialTargetContrast'][:ntrials]
optoChan = behavData['trialOptoChan'][:ntrials]
optoOnset = behavData['trialOptoOnset'][:ntrials]
rewardDir = behavData['trialRewardDir'][:ntrials]
response = behavData['trialResponse'][:ntrials]
responseDir = behavData['trialResponseDir'][:ntrials]

behavData.close()

goLeft = rewardDir==-1
goRight = rewardDir==1
catch = np.isnan(rewardDir)
noOpto = np.isnan(optoOnset)
optoLeft = optoChan[:,0] & ~optoChan[:,1]
optoRight = ~optoChan[:,0] & optoChan[:,1]
optoBoth = optoChan[:,0] & optoChan[:,1]

fig = plt.figure(figsize=(10,6))
gs = matplotlib.gridspec.GridSpec(2,2)
x = np.arange(4)
for j,contrast in enumerate([c for c in np.unique(targetContrast) if c>0]):
    for i,ylbl in enumerate(('Response Rate','Fraction Correct')):
        ax = fig.add_subplot(gs[i,j])
        for trials,trialLabel,clr,ty in zip((catch,goLeft,goRight),('catch','stim right (go left)','stim left (go right)'),'kbr',(1.05,1.1,1.15)):
            n = []
            y = []
            for opto in (noOpto,optoLeft,optoRight,optoBoth):
                ind = trials & opto
                if trialLabel != 'catch':
                    ind = trials & opto & (targetContrast==contrast)
                r = ~np.isnan(responseDir[ind])
                if ylbl=='Response Rate':
                    n.append(np.sum(ind))
                    y.append(r.sum()/n[-1])
                else:
                    n.append(r.sum())
                    if trialLabel=='catch':
                        y.append(np.nan)
                    else:
                        y.append(np.sum(r & (response[ind]==1))/n[-1])
            ax.plot(x,y,clr,marker='o',mec=clr,mfc='none',label=trialLabel)
            for tx,tn in zip(x,n):
                fig.text(tx,ty,str(tn),color=clr,transform=ax.transData,va='bottom',ha='center',fontsize=8)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(x)
        xticklabels = ('no\nopto','opto\nleft','opto\nright','opto\nboth') if i==1 else []
        ax.set_xticklabels(xticklabels)
        ax.set_xlim([-0.5,3.5])
        ax.set_ylim([0,1.05])
        if j==0:
            ax.set_ylabel(ylbl)
        if i==1 and j==0:
            ax.legend()
    tx = 0.3 if j==0 else 0.7
    fig.text(tx,0.99,'contrast '+str(contrast),va='top',ha='center')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for side,lbl,clr in zip((np.nan,-1,1),('no response','move left','move right'),'kbr'):
    n = []
    y = []
    for opto in (noOpto,optoLeft,optoRight,optoBoth):
        ind = catch & opto
        n.append(ind.sum())
        if np.isnan(side):
            y.append(np.sum(np.isnan(responseDir[ind]))/n[-1])
        else:
            y.append(np.sum(responseDir[ind]==side)/n[-1])
    ax.plot(x,y,clr,marker='o',mec=clr,mfc='none',label=lbl)
for tx,tn in zip(x,n):
    fig.text(tx,1.05,str(tn),color='k',transform=ax.transData,va='bottom',ha='center',fontsize=8)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(x)
ax.set_xticklabels(('no\nopto','opto\nleft','opto\nright','opto\nboth'))
ax.set_xlim([-0.5,3.5])
ax.set_ylim([0,1.05])
ax.set_ylabel('Fraction of catch trials')
ax.legend()
fig.text(0.525,0.99,'Catch trial movements',va='top',ha='center')

fig = plt.figure(figsize=(10,10))
gs = matplotlib.gridspec.GridSpec(8,2)
x = np.arange(4)
for j,contrast in enumerate([c for c in np.unique(targetContrast) if c>0]):
    for i,(trials,trialLabel) in enumerate(zip((goLeft,goRight,catch),('Right Stimulus','Left Stimulus','No Stimulus'))):
        if i<2 or j==0:
            ax = fig.add_subplot(gs[i*3:i*3+2,j])
            for resp,respLabel,clr,ty in zip((-1,1),('move left','move right'),'br',(1.05,1.1)):
                n = []
                y = []
                for opto in (noOpto,optoLeft,optoRight,optoBoth):
                    ind = trials & opto
                    if trialLabel != 'No Stimulus':
                        ind = trials & opto & (targetContrast==contrast)
                    n.append(ind.sum())
                    y.append(np.sum(responseDir[ind]==resp)/n[-1])
                ax.plot(x,y,clr,marker='o',mec=clr,mfc='none',label=respLabel)
            for tx,tn in zip(x,n):
                fig.text(tx,ty,str(tn),color='k',transform=ax.transData,va='bottom',ha='center',fontsize=8)
            title = trialLabel if trialLabel=='No Stimulus' else trialLabel+', Contrast '+str(contrast)
            fig.text(1.5,1.25,title,transform=ax.transData,va='bottom',ha='center',fontsize=10)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(x)
            xticklabels = ('no\nopto','opto\nleft','opto\nright','opto\nboth')# if i==2 else []
            ax.set_xticklabels(xticklabels)
            ax.set_xlim([-0.5,3.5])
            ax.set_ylim([0,1.05])
            if j==0:
                ax.set_ylabel('Fraction of trials')
            if i==0 and j==0:
                ax.legend(loc=(0.71,0.71))
                
                


