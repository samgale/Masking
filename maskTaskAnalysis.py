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
    t = bins[:-1]
    return counts,t


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

spikeData = {key: np.load(os.path.join(probeDataDir,'kilosort',key+'.npy')) for key in ('spike_clusters',
                                                                                        'spike_times',
                                                                                        'templates',
                                                                                        'spike_templates',
                                                                                        'channel_positions',
                                                                                        'amplitudes')}

clusterIDs = pd.read_csv(os.path.join(probeDataDir,'kilosort','cluster_KSLabel.tsv'),sep='\t')

unitIDs = np.unique(spikeData['spike_clusters'])

units = {}
for u in unitIDs:
    units[u] = {}
    units[u]['label'] = clusterIDs[clusterIDs['cluster_id']==u]['KSLabel'].tolist()[0]
    
    uind = np.where(spikeData['spike_clusters']==u)[0]
    
    units[u]['samples'] = spikeData['spike_times'][uind]
    
    #choose 1000 spikes with replacement, then average their templates together
    chosen_spikes = np.random.choice(uind,1000)
    chosen_templates = spikeData['spike_templates'][chosen_spikes].flatten()
    units[u]['template'] = np.mean(spikeData['templates'][chosen_templates],axis=0)
    
    peakChan = np.unravel_index(np.argmin(units[u]['template']),units[u]['template'].shape)[1]
    units[u]['peakChan'] = peakChan
    units[u]['position'] = spikeData['channel_positions'][peakChan]
    units[u]['amplitudes'] = spikeData['amplitudes'][uind]
    
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


totalChannels = 136
probeChannels = 128      

rawData = np.memmap(os.path.join(probeDataDir,'continuous.dat'),dtype='int16',mode='r')    
rawData = np.reshape(rawData,(int(rawData.size/totalChannels),-1)).T
totalSamples = rawData.shape[1]


analogInData = {name: rawData[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                          'photodiode',
                                                                          'rotaryEncoder',
                                                                          'cam1Saving',
                                                                          'cam2Saving',
                                                                          'cam1Exposure',
                                                                          'cam2Exposure',
                                                                          'led'))}

# get frame times and compare with psychopy frame intervals
frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))

behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')

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


#
goodUnits = np.array([u for u in units if units[u]['label']=='good'])
hasSpikes = np.array([units[u]['samples'].size/totalSamples*sampleRate for u in goodUnits]) > 0.1
goodUnits = goodUnits[hasSpikes]
goodUnits = goodUnits[np.argsort([units[u]['peakChan'] for u in goodUnits])]

peakToTrough = np.array([units[u]['peakToTrough'] for u in goodUnits])
fs = peakToTrough<=0.5
unitPos = np.array([units[u]['position'][1]/1000 for u in goodUnits])


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

optoOnsetToPlot = 0
optoOnsetTime = optoOnsetToPlot/frameRate
control = np.isnan(optoOnset)
opto = optoOnset==optoOnsetToPlot
targetOnly = (targetFrames>0) & (maskFrames==0)


psth = []
peakBaseRate = np.full(goodUnits.size,np.full)
meanBaseRate = peakBaseRate.copy()
transientOptoResp = peakBaseRate.copy()
sustainedOptoResp = peakBaseRate.copy()
sustainedOptoRate = peakBaseRate.copy()
preTime = 0.5
postTime = 0.5
trialTime = (openLoopFrames+responseWindowFrames)/frameRate
windowDur = preTime+trialTime+postTime
binSize = 0.005
trials = opto
for i,u in enumerate(goodUnits):
    spikeTimes = units[u]['samples']/sampleRate
    startTimes = frameSamples[stimStart[trials]]/sampleRate-preTime
    p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
    psth.append(p)
    peakBaseRate[i] = p[t<preTime].max()
    meanBaseRate[i] = p[t<preTime].mean()
    transientOptoResp[i] = p[(t>=preTime-binSize) & (t<preTime+trialTime)].max()-peakBaseRate[i]
    sustainedOptoRate[i] = p[(t>preTime+trialTime-0.25) & (t<preTime+trialTime)].mean()
    sustainedOptoResp[i] = sustainedOptoRate[i]-meanBaseRate[i]
psth = np.array(psth)
t -= preTime


fig = plt.figure(figsize=(8,8))
excit = sustainedOptoResp>0.5
inhib = ((sustainedOptoResp<0.1) & (transientOptoResp<0.5))
other = ~(excit | inhib)
gs = matplotlib.gridspec.GridSpec(3,2)
for i,j,clr,ind,lbl in zip((0,1,0,1,2),(0,0,1,1,1),'rbkkk',(fs,~fs,excit,inhib,other),('FS','RS','Excited','Inhibited','Other')):
    ax = fig.add_subplot(gs[i,j])
    ax.plot(t,psth[ind].mean(axis=0),clr)
    ylim = plt.get(ax,'ylim')
    poly = np.array([(optoOnsetTime,0),(trialTime+0.1,0),(trialTime,ylim[1]),(optoOnsetTime,ylim[1])])
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
        for ind,clr in zip((fs,~fs),'rb'):
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
            for x,lbl,clr in zip((0.25,0.75),('FS','RS'),'rb'):
                ax.text(x,1.05,lbl,transform=ax.transAxes,color=clr,fontsize=12,ha='center',va='bottom')
plt.tight_layout()



# testing
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
                
                
# masking opto
behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')

ntrials = behavData['trialEndFrame'].size
trialType = behavData['trialType'][:ntrials]
optoOnset = behavData['trialOptoOnset'][:ntrials]
rewardDir = behavData['trialRewardDir'][:ntrials]
response = behavData['trialResponse'][:ntrials]
responseDir = behavData['trialResponseDir'][:ntrials]

behavData.close()



# cam sync test
syncPath = fileIO.getFile('Select sync file',fileType='*.hdf5')
syncFile = h5py.File(syncPath,'r')
syncData = syncFile['AnalogInput']
syncSampleRate = syncData.attrs.get('sampleRate')
channelNames = syncData.attrs.get('channelNames')
cam1Saving = syncData[:,channelNames=='cam1Saving'][:,0]
cam1Exposure = syncData[:,channelNames=='cam1Exposure'][:,0]
syncFile.close()

saveFrameSamples = np.array(findSignalEdges(cam1Saving,edgeType='rising',thresh=0.5,refractory=2))

exposeFrameSamples = np.array(findSignalEdges(cam1Exposure,edgeType='rising',thresh=0.5,refractory=2))

print(saveFrameSamples.size,exposeFrameSamples.size)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(cam1Saving,color='k',label='saving')
ax.plot(cam1Exposure,color='0.5',label='exposure')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.legend()
plt.tight_layout()




