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



class MaskTaskData():
    
    def __init__(self):
        self.behav = False
        self.rf = False
        self.ephys = False
        self.frameDisplayLag = 2
        
        
    def loadBehavData(self,filePath=None):
        if filePath is None:
            self.behavDataPath = fileIO.getFile('Select behavior data file',fileType='*.hdf5')
        else:
            self.behavDataPath = filePath
        if len(self.behavDataPath)==0:
            return
        self.behav = True
        
        behavData = h5py.File(self.behavDataPath,'r')
        self.behavFrameIntervals = behavData['frameIntervals'][:]
        self.frameRate = round(1/np.median(self.behavFrameIntervals))
        if self.ephys and self.behavFrameIntervals.size+1>self.frameSamples.size:
            self.ntrials = np.sum(behavData['trialEndFrame'][:]<self.frameSamples.size)
        else:
            self.ntrials = behavData['trialEndFrame'].size
        self.quiescentFrames = behavData['quiescentFrames'][()]
        self.trialOpenLoopFrames = behavData['trialOpenLoopFrames'][:self.ntrials]
        assert(np.unique(self.trialOpenLoopFrames).size==1)
        self.openLoopFrames = self.trialOpenLoopFrames[0]
        self.responseWindowFrames = behavData['maxResponseWaitFrames'][()]
        self.wheelGain = behavData['wheelGain'][()]
        self.wheelRadius = behavData['wheelRadius'][()]
        self.wheelRewardDistance = behavData['wheelRewardDistance'][()]
        self.maxQuiescentMoveDist = behavData['maxQuiescentMoveDist'][()]
        self.deltaWheelPos = behavData['deltaWheelPos'][()]
        self.trialType = behavData['trialType'][:self.ntrials]
        self.stimStart = behavData['trialStimStartFrame'][:self.ntrials]
        self.targetContrast = behavData['trialTargetContrast'][:self.ntrials]
        self.targetFrames = behavData['trialTargetFrames'][:self.ntrials]
        self.maskContrast = behavData['trialMaskContrast'][:self.ntrials]
        self.maskFrames = behavData['trialMaskFrames'][:self.ntrials]
        self.maskOnset = behavData['trialMaskOnset'][:self.ntrials]
        self.rewardDir = behavData['trialRewardDir'][:self.ntrials]
        self.response = behavData['trialResponse'][:self.ntrials]
        self.responseDir = behavData['trialResponseDir'][:self.ntrials]
        self.responseFrame = behavData['trialResponseFrame'][:self.ntrials]
        self.optoChan = behavData['trialOptoChan'][:self.ntrials]
        self.optoOnset = behavData['trialOptoOnset'][:self.ntrials]
        
        self.findEngagedTrials()
        self.getWheelPos()
        self.findEarlyMoveTrials()
        self.calcReactionTime()
        
    def findEngagedTrials(self,engagedThresh=10):
        self.engaged = np.ones(self.ntrials,dtype=bool)
        trials = (obj.trialType!='catch') & np.isnan(self.optoOnset)
        for i in range(self.ntrials):
            r = self.responseDir[:i+1][trials[:i+1]]
            if len(r)>engagedThresh:
                if all(np.isnan(r[-engagedThresh:])):
                    self.engaged[i] = False
                    
    def getWheelPos(self,preFrames=0,postFrames=0):  
        deltaWheel = np.zeros((obj.ntrials,preFrames+self.openLoopFrames+obj.responseWindowFrames+postFrames))
        for i,s in enumerate(self.stimStart):
            d = self.deltaWheelPos[s-preFrames:s-preFrames+self.openLoopFrames+obj.responseWindowFrames+postFrames]
            deltaWheel[i,:len(d)] = d
        self.wheelPos = np.cumsum(deltaWheel,axis=1)
        self.wheelPos *= obj.wheelRadius
        
    def findEarlyMoveTrials(self,earlyMoveThresh=None):
        if earlyMoveThresh is None:
            earlyMoveThresh = self.maxQuiescentMoveDist
        obj.earlyMove = np.any(self.wheelPos[:,:self.openLoopFrames]>earlyMoveThresh,axis=1)
        
    def calcReactionTime(self,moveInitThresh=0.2):
        wp = self.wheelPos.copy()
        wp -= wp[:,self.openLoopFrames][:,None]
        wp[:,:self.openLoopFrames] = 0
        t = 1000/self.frameRate*(np.arange(wp.shape[1]))
        tinterp = np.arange(t[0],t[-1])
        self.reactionTime = np.full(obj.ntrials,np.nan)
        for i,w in enumerate(wp):
            winterp = np.interp(tinterp,t,np.absolute(w))
            respInd = np.where(winterp>=self.wheelRewardDistance)[0]
            if len(respInd)>0:
                initInd = np.where(winterp[:respInd[0]]<=moveInitThresh)[0]
                if len(initInd)>0:
                    self.reactionTime[i] = tinterp[initInd[-1]]+1
    
    
    def loadRFData(self):    
        self.rfDataPath = fileIO.getFile('Select rf mapping data file',fileType='*.hdf5')
        if len(self.rfDataPath)==0:
            return
        self.rf = True
        rfData = h5py.File(self.rfDataPath,'r')
        self.rfFrameIntervals = rfData['frameIntervals'][:]
        if self.behav is None:
            self.frameRate = round(1/np.median(self.rfFrameIntervals))
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
    
    
    def loadEphysData(self):
        self.datFilePath = fileIO.getFile('Select probe dat file',fileType='*.dat')
        if len(self.datDataPath)==0:
            return
        self.ephys = True
        probeDataDir = os.path.dirname(self.datFilePath)

        self.sampleRate = 30000
        totalChannels = 136
        probeChannels = 128      
        rawData = np.memmap(self.datFilePath,dtype='int16',mode='r')    
        rawData = np.reshape(rawData,(int(rawData.size/totalChannels),-1)).T
        analogInData = {name: rawData[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                                  'photodiode',
                                                                                  'rotaryEncoder',
                                                                                  'cam1Exposure',
                                                                                  'cam2Exposure',
                                                                                  'led1',
                                                                                  'led2'))}
        
        totalSamples = rawData.shape[1]
        self.frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))
        
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

        
    def saveToHdf5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
    
    
    def loadFromHdf5(self,filePath=None):
        fileIO.hdf5ToObj(self,filePath)
        


obj = MaskTaskData()

obj.saveToHdf5()

obj = MaskTaskData()
obj.loadFromHdf5()   


exps = []
for f in fileIO.getFiles('choose experiments',fileType='*.hdf5'):
    obj = MaskTaskData()
    obj.loadBehavData(f)
    exps.append(obj)

    
frameRate = 120
stimLabels = ('mask','targetOnly','maskOnly','catch')
rewardDir = (1,-1)
    


# masking
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
                    meanReacTime[n,i,j] = np.nanmean(obj.reactionTime[respTrials])
                    reacTime[n][stim][rd][mo] = obj.reactionTime[respTrials]
                    if stim in ('targetOnly','mask'):
                        correctTrials = obj.response[respTrials]==1
                        fracCorr[n,i,j] = correctTrials.sum()/respTrials.sum()
                        meanReacTimeCorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][correctTrials])
                        meanReacTimeIncorrect[n,i,j] = np.nanmean(obj.reactionTime[respTrials][~correctTrials])
                        reacTimeCorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][correctTrials]
                        reacTimeIncorrect[n][stim][rd][mo] = obj.reactionTime[respTrials][~correctTrials]
                    else:
                        break
                    
np.save(fileIO.saveFile(fileType='*.npy'),respRate)
np.save(fileIO.saveFile(fileType='*.npy'),fracCorr)

xticks = list(maskOnset[:-1]/frameRate*1000)+[67,83,100]
xticklabels = [str(int(round(x))) for x in xticks[:-3]]+['target\nonly','mask\nonly','no\nstimulus']
xlim = [8,108]

# single experiment
i = 0
for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0,1),None),('Response Rate','Fraction Correct','Reaction Time (ms)')):                  
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr in zip(data,'br'):
        ax.plot(xticks[:-3],d[:-3],'o',color=clr)
    ax.plot(xticks,np.nanmean(data,axis=0),'ko')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
# population
for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    meanLR = np.nanmean(data,axis=1)
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
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Mask onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
# reaction time on correct and incorrect trials
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for data,clr,lbl in zip((meanReacTimeCorrect,meanReacTimeIncorrect),('k','0.5'),('correct','incorrect')):
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
ax.set_xlim([150,650])
ax.set_ylim([0,1.02])
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=8,loc='upper left')

ax = fig.add_subplot(2,1,2)
for p,clr in zip(pc,clrs):
    ax.plot(bins[:-1]+binWidth/2,p,color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',right=False)
ax.set_xlim([150,650])
ax.set_ylim([0,1.02])
ax.set_xlabel('Reaction Time (ms)')
ax.set_ylabel('Probability Correct')
plt.tight_layout()


# opto masking
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

xticks = list(optoOnset[:-1]/frameRate*1000)+[117]
xticklabels = [str(int(round(x))) for x in xticks[:-1]]+['no\nopto']

for data,ylim,ylabel in zip((respRate,fracCorr,meanReacTime),((0,1),(0.4,1),None),('Response Rate','Fraction Correct','Mean reaction time (ms)')):        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(stim,stimLbl,clr) in enumerate(zip(stimLabels,('target + mask','target only','mask only','no stim'),'bckm')):
        meanLR = np.nanmean(data[:,i],axis=1)
        mean = np.nanmean(meanLR,axis=0)
        sem = np.nanstd(meanLR,axis=0)/(meanLR.shape[0]**0.5)
        if data is fracCorr:
            if stim=='targetOnly':
                firstValid = 3
            elif stim=='mask':
                firstValid = 2
            else:
                firstValid = 0
            lbls = ('response rate not above chance','response rate above chance') if stim=='maskOnly' else (None,None)
            ax.plot(xticks[:firstValid],mean[:firstValid],'o',ms=8,mec=clr,mfc='none',label=lbls[0])
            ax.plot(xticks[firstValid:-1],mean[firstValid:-1],'o',ms=8,mec=clr,mfc=clr,label=lbls[1])
        lbl = stimLbl if data is respRate else None
        ax.plot(xticks[:-1],mean[:-1],color=clr)
        ax.plot(xticks[-1],mean[-1],'o',ms=8,color=clr,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim([25,125])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Opto onset relative to target onset (ms)')
    ax.set_ylabel(ylabel)
    if data is not meanReacTime:
        ax.legend()
    plt.tight_layout()

 


# unilateral opto (old)
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
                
                


