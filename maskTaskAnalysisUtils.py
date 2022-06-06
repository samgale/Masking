# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:48:13 2021

@author: svc_ccg
"""


import os
import time
import h5py
import numpy as np
import pandas as pd
import scipy.signal
import scipy.ndimage
import scipy.optimize
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from numba import njit
import fileIO



def loadDatData(filePath,mode='r'):
    totalChannels = 136
    probeChannels = 128      
    data = np.memmap(filePath,dtype='int16',mode=mode)    
    data = np.reshape(data,(int(data.size/totalChannels),-1)).T
    analogInData = {name: data[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                           'photodiode',
                                                                           'rotaryEncoder',
                                                                           'cam1Exposure',
                                                                           'cam2Exposure',
                                                                           'led1',
                                                                           'led2'))}
    return data[:probeChannels],analogInData


def filterDatData(filePath,highpass=300,commonRef=True,ledArtifactDur=6):
    t = time.perf_counter()
    
    probeData,analogInData = loadDatData(filePath,mode='r+')
    sampleRate = 30000
    totalSamples = probeData.shape[1]
    
    # mask led artifacts
    if ledArtifactDur:
        led1Onsets,led2Onsets = [np.array(findSignalEdges(analogInData[ch],edgeType='rising',thresh=5000,refractory=5)) for ch in ('led1','led2')]
        ledOnsets = np.union1d(led1Onsets,led2Onsets).astype(int)
        x = np.arange(ledArtifactDur)
        for i in ledOnsets-1:
            for ch in probeData:
                if i < totalSamples-ledArtifactDur:
                    ch[i:i+ledArtifactDur] = np.interp(x,[0,ledArtifactDur],ch[[i,i+ledArtifactDur]])
                else:
                    ch[i:] = ch[i]
        print('masked '+str(len(ledOnsets))+' led arftifacts')
    
    if highpass or commonRef:
        if highpass:
            Wn = highpass/(sampleRate/2) # cutoff freq normalized to nyquist
            b,a = scipy.signal.butter(2,Wn,btype='highpass')
        
        chunkSamples = int(15*sampleRate)
        offset = 0
        while offset < totalSamples:
            d = probeData[:,offset:offset+chunkSamples]
            
            # highpass filter
            if highpass:
                d[:,:] = scipy.signal.filtfilt(b,a,d,axis=1)
            
            # common reference median filter
            if commonRef:
                d -= np.median(d,axis=0).astype(d.dtype)
    
            print('filtered '+str(offset)+' of '+str(totalSamples)+' samples')
            offset += chunkSamples
    
    # flush results (overwrites existing data)
    print('flushing to disk')
    del(probeData)
    del(analogInData)
    
    print('completed in '+str(time.perf_counter()-t)+' s')
    

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


@njit
def findSpikes(data,negThresh,posThresh):
    spikes = []    
    searchStart = 0
    while searchStart < data.size:
        spikeBegin,spikeEnd = findNextSpike(data[searchStart:],negThresh,posThresh)
        if spikeBegin is None:
            break
        else:
            spikes.append(searchStart+spikeBegin)
            searchStart += spikeEnd
    return spikes

@njit
def findNextSpike(data,negThresh,posThresh):
    for i,v in enumerate(data.flat):
        if v < negThresh:
            for j,vi in enumerate(data[i:].flat):
                if vi > posThresh:
                    return i,i+j
    return None,None


def getPsth(spikes,startTimes,windowDur,binSize=0.01,avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts, bins[:-1]+binSize/2


def getSdf(spikes,startTimes,windowDur,sampInt=0.001,filt='exponential',filtWidth=0.005,avg=True):
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
    
    
def getSyncData():
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
    

def fitCurve(func,x,y,initGuess=None,bounds=None):
    return scipy.optimize.curve_fit(func,x,y,p0=initGuess,bounds=bounds)[0]
    

def calcLogisticDistrib(x,a,b,m,s):
    # a: amplitude, b: offset, m: x at 50% max y, s: scale
    return a * (1 / (1 + np.exp(-(x - m) / s))) + b

def inverseLogistic(y,a,b,m,s):
    return m - s * np.log((a / (y - b)) - 1)


def calcWeibullDistrib(x,a,b,j,k):
    # a: amplitude, b: offset, j: shape, k: scale
    return a * (1 - np.exp(-(x / j) ** k)) + b

def inverseWeibull(y,a,b,j,k):
    return j * (-np.log(1 - ((y - b) / a))) ** (1/k)
    

class MaskTaskData():
    
    def __init__(self):
        self.behav = False
        self.rf = False
        self.ephys = False
        self.frameDisplayLag = 2
        self.earlyMoveFrames = 15
        
        
    def loadBehavData(self,filePath=None):
        if filePath is None:
            self.behavDataPath = fileIO.getFile('Select behavior data file',fileType='*.hdf5')
        else:
            self.behavDataPath = filePath
        if len(self.behavDataPath)==0:
            return
        self.behav = True
        print('\n'+self.behavDataPath)
        
        behavData = h5py.File(self.behavDataPath,'r')
        self.rigName = behavData['rigName'][()]
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
        self.trialStartFrame = behavData['trialStartFrame'][:self.ntrials]
        self.trialEndFrame = behavData['trialEndFrame'][:self.ntrials]
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
        if 'keyPressFrames' in behavData:
            self.keyPressFrames = behavData['keyPressFrames'][:]
            self.keysPressed = behavData['keysPressed'][:]
        if 'showVisibilityRating' in behavData and behavData['showVisibilityRating'][()]:
            self.visRating = behavData['visRating'][:self.ntrials]
            self.visRatingScore = np.zeros(self.visRating.size)
            self.visRatingScore[['1' in v for v in self.visRating]] = -1
            self.visRatingScore[['3' in v for v in self.visRating]] = 1
            self.visRatingStartFrame = behavData['visRatingStartFrame'][:self.ntrials]
            self.visRatingEndFrame = behavData['visRatingEndFrame'][:self.ntrials]
        self.useContrastStaircase = behavData['useContrastStaircase'][()] if 'useContrastStaircase' in behavData else False
        
        self.findLongFrameTrials()
        self.findEngagedTrials()
        self.getWheelPos()
        self.findEarlyMoveTrials()
        self.calcReactionTime()
        
    
    def findLongFrameTrials(self):
        self.longFrameTrials = np.zeros(self.ntrials,dtype=bool)
        self.targetDur = np.full(self.ntrials,np.nan)
        self.maskOnsetDur = self.targetDur.copy()
        self.optoOnsetDur = self.targetDur.copy()
        tol = 0.5/self.frameRate
        for i,s in enumerate(self.stimStart):
            if self.trialType[i] in ('targetOnly','targetOnlyOpto','mask','maskOpto'):
                self.targetDur[i] = self.behavFrameIntervals[s:s+self.targetFrames[i]].sum()
                if  self.targetDur[i] > self.targetFrames[i]/self.frameRate+tol:
                    self.longFrameTrials[i] = True
            if self.trialType[i] in ('mask','maskOpto'):
                self.maskOnsetDur[i] = self.behavFrameIntervals[s:s+self.maskOnset[i]].sum()
                if self.maskOnsetDur[i] > self.maskOnset[i]/self.frameRate+tol:
                    self.longFrameTrials[i] = True
            if not np.isnan(self.optoOnset[i]):
                self.optoOnsetDur[i] = self.behavFrameIntervals[s:s+int(self.optoOnset[i])].sum()
                if self.optoOnsetDur[i] > self.optoOnset[i]/self.frameRate+tol:
                    self.longFrameTrials[i] = True
        print(str(round(100*np.sum(self.behavFrameIntervals > 1/self.frameRate+tol)/self.behavFrameIntervals.size,2))+' % frames long')
        print(str(self.longFrameTrials.sum())+' / '+str(self.ntrials)+' trials had long frames')
    
    
    def findEngagedTrials(self,engagedThresh=10):
        self.engaged = np.ones(self.ntrials,dtype=bool)
        trials = (self.trialType!='catch') & np.isnan(self.optoOnset)
        for i in range(self.ntrials):
            r = self.responseDir[:i+1][trials[:i+1]]
            if len(r)>engagedThresh:
                if all(np.isnan(r[-engagedThresh:])):
                    self.engaged[i] = False
        print(str(self.engaged.sum())+' / '+str(self.ntrials)+' trials engaged')
    
                
    def getWheelPos(self,preFrames=0,postFrames=0):  
        deltaWheel = np.zeros((self.ntrials,preFrames+self.openLoopFrames+self.responseWindowFrames+postFrames))
        for i,s in enumerate(self.stimStart):
            d = self.deltaWheelPos[s-preFrames:s-preFrames+self.openLoopFrames+self.responseWindowFrames+postFrames]
            deltaWheel[i,:len(d)] = d
        self.wheelPos = np.cumsum(deltaWheel,axis=1)
        self.wheelPos *= self.wheelRadius
    
    
    def findEarlyMoveTrials(self,earlyMoveThresh=None):
        if earlyMoveThresh is None:
            earlyMoveThresh = self.maxQuiescentMoveDist
        self.earlyMove = np.any(self.wheelPos[:,:self.earlyMoveFrames]>earlyMoveThresh,axis=1)
        print(str(self.earlyMove.sum())+' / '+str(self.ntrials)+' trials early move')
    
    
    def calcReactionTime(self,moveInitThresh=0.2):
        self.reactionTime = np.full(self.ntrials,np.nan)
        self.movementVelocity = np.full(self.ntrials,np.nan)
        if self.rigName == 'human':
            for i,(s,r) in enumerate(zip(self.stimStart+self.frameDisplayLag,self.responseFrame)):
                self.reactionTime[i] = self.behavFrameIntervals[s+1:r].sum()*1000
            if hasattr(self,'visRating'):
                self.visRatingReactionTime = np.full(self.ntrials,np.nan)
                for i,(s,r) in enumerate(zip(self.visRatingStartFrame+self.frameDisplayLag,self.visRatingEndFrame)):
                    self.visRatingReactionTime[i] = self.behavFrameIntervals[s+1:r].sum()*1000 
        else:
            wp = self.wheelPos-self.wheelPos[:,self.earlyMoveFrames][:,None]
            wp[:,:self.earlyMoveFrames] = 0
            for i,(w,s) in enumerate(zip(wp,self.stimStart+self.frameDisplayLag)):
                frameIntervals = self.behavFrameIntervals[s:s+w.size]
                frameIntervals[0] = 0
                t = np.cumsum(frameIntervals)
                t *= 1000
                tinterp = np.arange(t[-1])
                winterp = np.interp(tinterp,t,np.absolute(w[:t.size]))
                respInd = np.where(winterp>=self.wheelRewardDistance)[0]
                if len(respInd)>0:
                    belowThresh = np.where(winterp[:respInd[0]]<moveInitThresh)[0]
                    if len(belowThresh)>0:
                        initInd = belowThresh[-1]+1
                        self.reactionTime[i] = tinterp[initInd]
                        self.movementVelocity[i] = 1000*(self.wheelRewardDistance-moveInitThresh)/(tinterp[respInd[0]]-tinterp[initInd])
    
    
    def loadRFData(self,filePath=None):
        if filePath is None:
            self.rfDataPath = fileIO.getFile('Select rf mapping data file',fileType='*.hdf5')
        else:
            self.rfDataPath = filePath
        if len(self.rfDataPath)==0:
            return
        self.rf = True
        rfData = h5py.File(self.rfDataPath,'r')
        self.rfFrameIntervals = rfData['frameIntervals'][:]
        if not self.behav:
            self.frameRate = round(1/np.median(self.rfFrameIntervals))
        if 'stimStartFrame' in rfData:
            self.rfStimStart = rfData['stimStartFrame'][:-1]
        else:
            trialStartFrame = np.concatenate(([0],np.cumsum(rfData['preFrames']+rfData['trialStimFrames'][:-1]+rfData['postFrames'])))
            self.rfStimStart = trialStartFrame+rfData['preFrames']
        self.rfStimStart += self.frameSamples.size-(self.rfFrameIntervals.size+1)
        rfTrials = self.rfStimStart.size
        self.rfStimPos = rfData['trialGratingCenter'][:rfTrials]
        self.rfStimContrast = rfData['trialGratingContrast'][:rfTrials]
        self.rfOris = rfData['gratingOri'][:rfTrials]
        self.rfStimOri = rfData['trialGratingOri'][:rfTrials]
        self.rfStimFrames = rfData['trialStimFrames'][:rfTrials]
    
    
    def loadEphysData(self,led=False):
        self.datFilePath = fileIO.getFile('Select probe dat file',fileType='*.dat')
        if len(self.datFilePath)==0:
            return
        self.ephys = True

        probeData,analogInData = loadDatData(self.datFilePath)
        
        self.sampleRate = 30000
        self.totalSamples = probeData.shape[1]
        
        self.frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))
        
        if led:
            self.led1Onsets,self.led2Onsets = [np.array(findSignalEdges(analogInData[ch],edgeType='rising',thresh=5000,refractory=5)) for ch in ('led1','led2')]
        
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
        
    
    def loadKilosortData(self,dirPath=None):
        if dirPath is None:
            self.kilosortDirPath = fileIO.getDir('Select directory containing kilosort data')
        else:
            self.kilosortDirPath = dirPath
        if len(self.kilosortDirPath)==0:
            return
        kilosortData = {key: np.load(os.path.join(self.kilosortDirPath,key+'.npy')) for key in ('spike_clusters',
                                                                                                'spike_times',
                                                                                                'templates',
                                                                                                'spike_templates',
                                                                                                'channel_positions',
                                                                                                'amplitudes')}
        clusterIDs = pd.read_csv(os.path.join(self.kilosortDirPath,'cluster_KSLabel.tsv'),sep='\t')
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
        
        self.sortedUnits = np.array(list(self.units.keys()))[np.argsort([self.units[u]['peakChan'] for u in self.units])]
        self.findIsiViolations()
        self.getGoodUnits()
        
    
    def findIsiViolations(self,minIsi=0,refracPeriod=0.0015):
        totalTime = self.totalSamples/self.sampleRate
        for u in self.units:
            spikeTimes = self.units[u]['samples']/self.sampleRate
            duplicateSpikes = np.where(np.diff(spikeTimes)<=minIsi)[0]+1
            spikeTimes = np.delete(spikeTimes,duplicateSpikes)
            isis = np.diff(spikeTimes)
            numSpikes = len(spikeTimes)
            numViolations = sum(isis<refracPeriod)
            violationTime = 2*numSpikes*(refracPeriod-minIsi)
            violationRate = numViolations/violationTime
            totalRate = numSpikes/totalTime
            self.units[u]['fpRate'] = violationRate/totalRate
   
         
    def getGoodUnits(self,fpThresh=0.5,minRate=0.1):
        self.goodUnits = [u for u in self.sortedUnits if self.units[u]['label']!='noise' and self.units[u]['fpRate']<fpThresh and len(self.units[u]['samples'])/(self.totalSamples/self.sampleRate)>minRate]
        
    def saveToHdf5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
    
    
    def loadFromHdf5(self,filePath=None):
        fileIO.hdf5ToObj(self,filePath)


