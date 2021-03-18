# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import os
import pickle
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


def loadDatData(filePath):
    totalChannels = 136
    probeChannels = 128      
    rawData = np.memmap(filePath,dtype='int16',mode='r')    
    rawData = np.reshape(rawData,(int(rawData.size/totalChannels),-1)).T
    analogInData = {name: rawData[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                              'photodiode',
                                                                              'rotaryEncoder',
                                                                              'cam1Exposure',
                                                                              'cam2Exposure',
                                                                              'led1',
                                                                              'led2'))}
    return rawData,analogInData


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




class MaskingEphys():
    
    def __init__(self):
        pass
    
    def loadData(self):
        # get probe data
        self.datFilePath = fileIO.getFile('Select probe dat file',fileType='*.dat')
        probeDataDir = os.path.dirname(self.datFilePath)

        self.sampleRate = 30000
        
        rawData,analogInData = loadDatData(self.datFilePath)
        totalSamples = rawData.shape[1]
        
        kilosortData = {key: np.load(os.path.join(probeDataDir,'kilosort',key+'.npy')) for key in ('spike_clusters',
                                                                                                   'spike_times',
                                                                                                   'templates',
                                                                                                   'spike_templates',
                                                                                                   'channel_positions',
                                                                                                   'amplitudes')}

        clusterIDs = pd.read_csv(os.path.join(probeDataDir,'kilosort','cluster_KSLabel.tsv'),sep='\t')
        
        unitIDs = np.unique(kilosortData['spike_clusters'])
        
        self.units = {}
        for u in unitIDs:
            uind = np.where(kilosortData['spike_clusters']==u)[0]
            
            u = str(u)
            
            self.units[u] = {}
            self.units[u]['label'] = clusterIDs[clusterIDs['cluster_id']==int(u)]['KSLabel'].tolist()[0]
            self.units[u]['samples'] = kilosortData['spike_times'][uind].flatten()
            
            #choose 1000 spikes with replacement, then average their templates together
            chosen_spikes = np.random.choice(uind,1000)
            chosen_templates = kilosortData['spike_templates'][chosen_spikes].flatten()
            self.units[u]['template'] = np.mean(kilosortData['templates'][chosen_templates],axis=0)
            
            peakChan = np.unravel_index(np.argmin(self.units[u]['template']),self.units[u]['template'].shape)[1]
            self.units[u]['peakChan'] = peakChan
            self.units[u]['position'] = kilosortData['channel_positions'][peakChan]
            self.units[u]['amplitudes'] = kilosortData['amplitudes'][uind]
            
            template = self.units[u]['template'][:,peakChan]
            if any(np.isnan(template)):
                self.units[u]['peakToTrough'] = np.nan
            else:
                peakInd = np.argmin(template)
                self.units[u]['peakToTrough'] = np.argmax(template[peakInd:])/(self.sampleRate/1000)
            
            #check if this unit is noise
            tempNorm = template/np.max(np.absolute(template))
            self.units[u]['normTempIntegral'] = tempNorm.sum()
            if abs(tempNorm.sum())>4:
                self.units[u]['label'] = 'noise'
                
        self.goodUnits = np.array([u for u in self.units if self.units[u]['label']=='good'])

        #epochs = 4
        #epochSamples = totalSamples/epochs
        #hasSpikes = np.ones(len(self.goodUnits),dtype=bool)
        #for i in range(epochs):
        #    hasSpikes = hasSpikes & (np.array([np.sum((self.units[u]['samples']>=i*epochSamples) & (self.units[u]['samples']<i*epochSamples+epochSamples))/epochSamples*sampleRate for u in self.goodUnits]) > 0.1)
        
        hasSpikes = [(self.units[u]['samples'].size/totalSamples*self.sampleRate)>0.1 for u in self.goodUnits]
        self.goodUnits = self.goodUnits[hasSpikes]
        
        self.goodUnits = self.goodUnits[np.argsort([self.units[u]['peakChan'] for u in self.goodUnits])]

        self.peakToTrough = np.array([self.units[u]['peakToTrough'] for u in self.goodUnits])
        
        self.unitPos = np.array([self.units[u]['position'][1]/1000 for u in self.goodUnits])

        # get behavior and rf mapping data
        totalFrames = 0
        self.frameRate = None
        
        self.behavDataPath = fileIO.getFile('Select behavior data file',fileType='*.hdf5')
        if len(self.behavDataPath)>0:
            behavData = h5py.File(self.behavDataPath,'r')
            self.behavFrameIntervals = behavData['frameIntervals'][:]
            totalFrames += self.behavFrameIntervals.size+1
            print(str(self.behavFrameIntervals.size+1)+' behavior frames')
            self.frameRate = round(1/np.median(self.behavFrameIntervals))
        
        self.rfDataPath = fileIO.getFile('Select rf mapping data file',fileType='*.hdf5')
        if len(self.rfDataPath)>0:
            rfData = h5py.File(self.rfDataPath,'r')
            self.rfFrameIntervals = rfData['frameIntervals'][:]
            totalFrames += self.rfFrameIntervals.size+1
            print(str(self.rfFrameIntervals.size+1)+' rf frames')
            if self.frameRate is None:
                self.frameRate = round(1/np.median(self.rfFrameIntervals))
        
        # get frame times and compare with psychopy frame intervals
        self.frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))
        
        print(str(totalFrames)+' total frames')
        print(str(self.frameSamples.size)+' frame signals')
        
        if len(self.behavDataPath)>0:
            if self.behavFrameIntervals.size+1>self.frameSamples.size:
                self.ntrials = np.sum(behavData['trialEndFrame'][:]<self.frameSamples.size)
            else:
                self.ntrials = behavData['trialEndFrame'].size
            self.stimStart = behavData['trialStimStartFrame'][:self.ntrials]
            self.trialOpenLoopFrames = behavData['trialOpenLoopFrames'][:self.ntrials]
            if np.unique(self.trialOpenLoopFrames).size>1:
                print('multiple values of open loop frames')
            self.openLoopFrames = self.trialOpenLoopFrames[0]
            self.responseWindowFrames = behavData['maxResponseWaitFrames'][()]
            self.trialType = behavData['trialType'][:self.ntrials]
            self.targetContrast = behavData['trialTargetContrast'][:self.ntrials]
            self.targetFrames = behavData['trialTargetFrames'][:self.ntrials]
            self.maskContrast = behavData['trialMaskContrast'][:self.ntrials]
            self.maskFrames = behavData['trialMaskFrames'][:self.ntrials]
            self.maskOnset = behavData['trialMaskOnset'][:self.ntrials]
            self.rewardDir = behavData['trialRewardDir'][:self.ntrials]
            self.response = behavData['trialResponse'][:self.ntrials]
            self.responseDir = behavData['trialResponseDir'][:self.ntrials]
            self.optoChan = behavData['trialOptoChan'][:self.ntrials]
            self.optoOnset = behavData['trialOptoOnset'][:self.ntrials]
            
        if len(self.rfDataPath)>0:
            if 'stimStartFrame' in rfData:
                self.rfStimStart = rfData['stimStartFrame'][:-1]
            else:
                trialStartFrame = np.concatenate(([0],np.cumsum(rfData['preFrames']+rfData['trialStimFrames'][:-1]+rfData['postFrames'])))
                self.rfStimStart = trialStartFrame+rfData['preFrames']
            self.rfStimStart += obj.frameSamples.size-(obj.rfFrameIntervals.size+1)
            rfTrials = self.rfStimStart.size
            self.rfStimPos = rfData['trialGratingCenter'][:rfTrials]
            self.rfStimContrast = rfData['trialGratingContrast'][:rfTrials]
            self.rfOris = rfData['gratingOri'][:rfTrials]
            self.rfStimOri = rfData['trialGratingOri'][:rfTrials]
            self.rfStimFrames = rfData['trialStimFrames'][:rfTrials]
        
        # check frame display lag
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        samples = np.arange(self.frameSamples[0]-1500,self.frameSamples[0]+3001)
        t = (samples-self.frameSamples[0])/self.sampleRate
        ax.plot(t,analogInData['vsync'][samples],color='k',label='vsync')
        ax.plot(t,analogInData['photodiode'][samples],color='0.5',label='photodiode')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Time from first frame (s)')
        ax.legend()
        plt.tight_layout()
        
        self.frameDisplayLag = 2
        
    def saveToHdf5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
    
    def loadFromHdf5(self,filePath=None):
        fileIO.hdf5ToObj(self,filePath)
        

obj = MaskingEphys()
obj.loadData()

obj.saveToHdf5()

obj = MaskingEphys()
obj.loadFromHdf5()   


exps = []
while True:
    f = fileIO.getFile('choose masking ephys experiment',fileType='*.hdf5')
    if f!='':
        obj = MaskingEphys()
        obj.loadFromHdf5(f)
        exps.append(obj)
    else:
        break
    

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


# plot response to optogenetic stimuluation during catch trials
psth = []
peakBaseRate = np.full(sum(obj.goodUnits.size for obj in exps),np.nan)
meanBaseRate = peakBaseRate.copy()
transientOptoResp = peakBaseRate.copy()
sustainedOptoResp = peakBaseRate.copy()
sustainedOptoRate = peakBaseRate.copy()
preTime = 0.5
postTime = 0.5
trialTime = [(obj.openLoopFrames+obj.responseWindowFrames)/obj.frameRate for obj in exps]
assert(all([t==trialTime[0] for t in trialTime]))
trialTime = trialTime[0]
windowDur = preTime+trialTime+postTime
binSize = 1/exps[0].frameRate
i = 0
for obj in exps:
    optoOnsetToPlot = np.nanmin(obj.optoOnset)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        p = []
        for onset in [optoOnsetToPlot]: #np.unique(optoOnset[~np.isnan(optoOnset)]):
            trials = (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            s,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
            p.append(s)
        p = np.mean(np.concatenate(p),axis=0)
        psth.append(p)
        peakBaseRate[i] = p[t<preTime].max()
        meanBaseRate[i] = p[t<preTime].mean()
        transientOptoResp[i] = p[(t>=preTime) & (t<preTime+0.25)].max()-peakBaseRate[i]
        sustainedOptoRate[i] = p[(t>preTime+trialTime-0.35) & (t<preTime+trialTime-0.25)].mean()
        sustainedOptoResp[i] = sustainedOptoRate[i]-meanBaseRate[i]
        i += 1
t -= preTime
psth = np.array(psth)

fs = np.concatenate([obj.peakToTrough<=0.5 for obj in exps])

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
    poly = np.array([(0,0),(trialTime-optoOnsetToPlot/obj.frameRate+0.1,0),(trialTime,ylim[1]),(0,ylim[1])])
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
for j,(xdata,xlbl) in enumerate(zip((obj.peakToTrough,obj.unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
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
respThresh = 5 # stdev
stimLabels = ('targetOnly','maskOnly','mask')
respLabels = ('all','go','nogo')
cellTypeLabels = ('all','FS','RS')
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/obj.frameRate
ntrials = {stim: {side: {resp: {} for resp in respLabels} for side in ('left','right')} for stim in stimLabels}
psth = {cellType: {stim: {side: {resp: {} for resp in respLabels} for side in ('left','right')} for stim in stimLabels} for cellType in cellTypeLabels}
hasResp = copy.deepcopy(psth)
peakResp = copy.deepcopy(psth)
timeToPeak = copy.deepcopy(psth)
timeToFirstSpike = copy.deepcopy(psth)
for stim in stimLabels:
    for rd,side in zip((1,-1),('left','right')):
        for resp in respLabels:
            for obj in exps:
                fs = obj.peakToTrough<=0.5
                stimTrials = obj.trialType==stim if stim=='maskOnly' else (obj.trialType==stim) & (obj.rewardDir==rd)
                if resp=='all':
                    respTrials = np.ones(obj.ntrials,dtype=bool)
                else:
                    respTrials = ~np.isnan(obj.responseDir) if resp=='go' else np.isnan(obj.responseDir)
                for mo in np.unique(obj.maskOnset[stimTrials]):
                    moTrials = obj.maskOnset==mo
                    trials = stimTrials & respTrials & (obj.maskOnset==mo)
                    if mo not in ntrials[stim][side][resp]:
                        ntrials[stim][side][resp][mo] = 0
                    ntrials[stim][side][resp][mo] += trials.sum()
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
                        if mo not in psth[cellType][stim][side][resp]:
                            psth[cellType][stim][side][resp][mo] = []
                            hasResp[cellType][stim][side][resp][mo] = []
                            peakResp[cellType][stim][side][resp][mo] = []
                            timeToPeak[cellType][stim][side][resp][mo] = []
                            timeToFirstSpike[cellType][stim][side][resp][mo] = []
                        for u in obj.goodUnits[ct]:
                            spikeTimes = obj.units[u]['samples']/obj.sampleRate
                            p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                            t -= preTime
                            analysisWindow = (t>0.03) & (t<0.15)
                            p -= p[t<0].mean()
                            psth[cellType][stim][side][resp][mo].append(p)
                            hasResp[cellType][stim][side][resp][mo].append(p[analysisWindow].max() > respThresh*p[t<0].std())
                            peakResp[cellType][stim][side][resp][mo].append(p[analysisWindow].max())
                            timeToPeak[cellType][stim][side][resp][mo].append(t[np.argmax(p[analysisWindow])+np.where(analysisWindow)[0][0]])
                            lat = []
                            for st in startTimes:
                                firstSpike = np.where((spikeTimes > st+0.03) & (spikeTimes < st+0.15))[0]
                                if len(firstSpike)>0:
                                    lat.append(spikeTimes[firstSpike[0]]-st)
                                else:
                                    lat.append(np.nan)
                            timeToFirstSpike[cellType][stim][side][resp][mo].append(np.nanmedian(lat))

respCells = {cellType: np.array(hasResp[cellType]['targetOnly']['right']['all'][0]) | np.array(hasResp[cellType]['maskOnly']['right']['all'][0]) for cellType in cellTypeLabels}


xlim = [-0.1,0.4]
for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
    if cellType!='all':
        continue
    axs = []
    ymin = ymax = 0
    for resp in respLabels:
        fig = plt.figure(figsize=(10,5))
        fig.text(0.5,0.99,cellType+' (n='+str(respCells[cellType].sum())+' cells)',ha='center',va='top',fontsize=12)
        for i,(rd,side) in enumerate(zip((1,-1),('left','right'))):
            ax = fig.add_subplot(1,2,i+1)
            axs.append(ax)
            for stim,clr in zip(stimLabels,('k','0.5','r')):
                stimTrials = obj.trialType==stim if stim=='maskOnly' else (obj.trialType==stim) & (obj.rewardDir==rd)
                mskOn = np.unique(obj.maskOnset[stimTrials])
#                mskOn = [mskOn[-1]]
                if stim=='mask' and len(mskOn)>1:
                    cmap = np.ones((len(mskOn),3))
                    cint = 1/len(mskOn)
                    cmap[:,1:] = np.arange(0,1.01-cint,cint)[:,None]
                else:
                    cmap = [clr]
                for mo,c in zip(mskOn,cmap):
                    p = np.array(psth[cellType][stim][side][resp][mo])[respCells[cellType]]
                    m = np.mean(p,axis=0)
                    s = np.std(p,axis=0)/(len(p)**0.5)
                    lbl = 'target+mask, SOA '+str(round(1000*mo/obj.frameRate,1))+' ms' if stim=='mask' else stim
                    rlbl = '' if resp=='all' else ' '+resp
                    lbl += ' ('+str(ntrials[stim][side][resp][mo])+rlbl+' trials)'
#                    tme = t+2/obj.frameRate if stim=='maskOnly' else t
                    ax.plot(t,m,color=c,label=lbl)
        #            ax.fill_between(t,m+s,m-s,color=c,alpha=0.25)
                    ymin = min(ymin,np.min(m[(t>=xlim[0]) & (t<=xlim[1])]))
                    ymax = max(ymax,np.max(m[(t>=xlim[0]) & (t<=xlim[1])]))
            for s in ('right','top'):
                ax.spines[s].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from stimulus onset (s)')
            if i==0:
                ax.set_ylabel('Response (spikes/s)')
            ax.legend(loc='upper left',frameon=False,fontsize=8)
            ax.set_title('target '+side)
    for ax in axs:
        ax.set_ylim([1.05*ymin,1.05*ymax])
    

fig = plt.figure(figsize=(10,6))
gs = matplotlib.gridspec.GridSpec(2,2)
for j,(xdata,xlbl) in enumerate(zip((obj.peakToTrough,obj.unitPos),('Spike peak to trough (ms)','Distance from probe tip (mm)'))):
    for i,stim in enumerate(stimLabels[:2]):
        ax = fig.add_subplot(gs[i,j])
        for ct,cellType,clr in zip((fs,~fs),('FS','RS'),'mg'):
            for r,mfc in zip((~respCells[cellType],respCells[cellType]),('none',clr)):
                ax.plot(xdata[ct][r],peakResp[cellType][stim]['right']['all'][0][r],'o',mec=clr,mfc=mfc)
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
    for r,mfc in zip((~respCells[cellType],respCells[cellType]),('none',clr)):
        x,y = [peakResp[cellType][stim]['right']['all'][0][r] for stim in stimLabels[:2]]
        if any(x) and any(y):
            amax = max(amax,x.max(),y.max())
            ax.plot(x,y,'o',mec=clr,mfc=mfc,label=cellType)
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


fig = plt.figure(figsize=(8,6))
gs = matplotlib.gridspec.GridSpec(2,2)
for j,cellType in enumerate(('FS','RS')):
    for i,(ydata,ylbl) in enumerate(zip((timeToPeak,timeToFirstSpike),('Time to peak (ms)','Time to first spike (ms)'))):
        ax = fig.add_subplot(gs[i,j])
        for x,stim in enumerate(stimLabels):
            y = 1000*np.array(ydata[cellType][stim]['right']['all'][0])[respCells[cellType]]
            ax.plot(x+np.zeros(len(y)),y,'o',mec='0.5',mfc='none')
            m = np.nanmean(y)
            s = np.nanstd(y)/(np.sum(~np.isnan(y))**0.5)
            ax.plot(x,m,'ko')
            ax.plot([x,x],[m-s,m+s],'k')
            ax.text(x+0.1,m,str(round(m,1)),transform=ax.transData,ha='left',va='center')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(len(stimLabels)))
        if i==0:
            ax.set_xticklabels([])
            ax.set_title(cellType)
        else:
            ax.set_xticklabels(('target only','mask only','SOA 16.7 ms'))
        ax.set_xlim([-0.5,2.5])
        ax.set_ylim([30,150])
        if j==0:
            ax.set_ylabel(ylbl)
plt.tight_layout()


popPsth = {stim: {hemi: {} for hemi in ('ipsi','contra')} for stim in stimLabels}
for cellType in ('all',):
    for stim in stimLabels:
        for side,hemi in zip(('left','right'),('ipsi','contra')):
            p = psth[cellType][stim][side]['all']
            for mo in p.keys():
                popPsth[stim][hemi][mo] = np.mean(np.array(p[mo])[respCells[cellType]],axis=0)
popPsth['t'] = t
            
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(popPsth,open(pkl,'wb'))



# plot response to visual stimuli with opto
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/obj.frameRate
optOn = list(np.unique(obj.optoOnset[~np.isnan(obj.optoOnset)]))+[np.nan]
cmap = np.zeros((len(optOn),3))
cint = 1/(len(optOn)-1)
cmap[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
cmap[:-1,2] = 1
fig = plt.figure(figsize=(6,10))
axs = []
naxs = 2+np.sum(np.unique(obj.maskOnset>0))
for stim in stimLabels:
    stimTrials = np.in1d(obj.trialType,(stim,stim+'Opto'))
    if stim!='maskOnly':
        stimTrials = stimTrials & (obj.rewardDir==-1)
    for mo in np.unique(obj.maskOnset[stimTrials]):
        ax = fig.add_subplot(naxs,1,len(axs)+1)
        axs.append(ax)
        moTrials = obj.maskOnset==mo
        for onset,clr in zip(optOn,cmap):
            trials = np.isnan(obj.optoOnset) if np.isnan(onset) else obj.optoOnset==onset
            trials = trials & stimTrials & moTrials
            startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
            psth = []
            for u in obj.goodUnits[~fs & inhib]:
                spikeTimes = obj.units[u]['samples']/obj.sampleRate
                p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                psth.append(p)
            t -= preTime
            m = np.mean(psth,axis=0)
            s = np.std(psth,axis=0)/(len(psth)**0.5)
            lbl = 'no opto' if np.isnan(onset) else str(int(round(1000*(onset-obj.frameDisplayLag)/obj.frameRate)))+' ms'
            ax.plot(t,m,color=clr,label=lbl)
#            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(-0.05,0.21,0.05))
        ax.set_xlim([-0.05,0.2])
        ax.set_ylabel('Spikes/s')
        title = 'SOA '+str(int(round(1000*mo/obj.frameRate)))+' ms' if stim=='mask' else stim
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


# rf mapping
azi,ele = [np.unique(p) for p in obj.rfStimPos.T]
ori = obj.rfOris
contrast = np.unique(obj.rfStimContrast)
dur = np.unique(obj.rfStimFrames)
binSize = 1/obj.frameRate
preTime = 0.1
postTime = 0.6
nbins = np.arange(0,preTime+postTime+binSize,binSize).size-1
rfMap = np.zeros((obj.goodUnits.size,nbins,ele.size,azi.size,ori.shape[0],contrast.size,dur.size))
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
                    for n,u in enumerate(obj.goodUnits):
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPSTH(spikeTimes,startTimes-preTime,preTime+postTime,binSize=binSize,avg=True)
                        t -= preTime
#                        p -= p[t<0].mean()
                        rfMap[n,:,i,j,k,l,m] = p


fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(ele.size,azi.size)
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        ax = fig.add_subplot(gs[ele.size-1-i,j])
        
        ax.plot(t,rfMap.mean(axis=(0,4,5,6))[:,i,j],'k')
        
#        for k,d in enumerate(dur):
#            ax.plot(t,rfMap.mean(axis=(0,4,5))[:,i,j,k],'k')
            
#        for k,c in enumerate(contrast):
#            ax.plot(t,rfMap.mean(axis=(0,4,6))[:,i,j,k],'k')
        
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==2].mean(axis=(0,4,5,6))[:,i,j],'k',label='17 ms stim')
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==6].mean(axis=(0,4,5,6))[:,i,j],'r',label='50 ms stim')
            
        # target
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==2].mean(axis=(0,5,6))[:,i,j,0],'k')
        
        # mask
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==6].mean(axis=(0,5,6))[:,i,j,4],'k')
        
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
        if i==ele.size-1 and j==azi.size-1:
            ax.legend(loc='upper right')
        ax.set_xticks([0,0.5])
        ax.set_yticks([0,10,20,30])
        ax.set_ylim([0,35])
plt.tight_layout()


plt.figure()
for i,o in enumerate(ori):
    plt.plot(t,rfMap.mean(axis=(0,2,3,5,6))[:,i],label=o)
plt.legend()




# unilateral opto
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
            ax.legend(fontsize='small', loc='best')
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


fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(8,1)
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
                ax.legend(fontsize='small', loc=(0.71,0.71))
                
                


