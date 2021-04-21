# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:44:29 2021

@author: svc_ccg
"""

import time
import h5py
import numpy as np
import scipy.signal
import matplotlib
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


def getPSTH(spikes,startTimes,windowDur,binSize=0.01,avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts, bins[:-1]+binSize/2


sampleRate = 30000
channelMap = np.array([16,112,111,14,110,15,109,12,108,13,107,10,106,11,105,8,104,9,103,6,102,7,101,4,100,5,99,2,98,3,97,96,1,95,62,94,63,93,60,92,61,91,58,90,59,89,56,88,57,87,55,86,54,85,53,84,52,83,51,82,50,81,49,80,48,47,78,46,79,45,76,44,77,43,75,42,74,41,73,40,72,39,71,38,70,37,69,36,68,35,67,34,66,33,65,32,64,31,127,30,126,29,125,28,124,27,123,26,122,25,121,24,120,23,119,22,118,21,117,20,116,19,115,18,114,17,113])-1
channelRange = [0,127]


# get data
datPath = fileIO.getFile('Select probe dat file',fileType='*.dat')
datData,analogInData = loadDatData(datPath)
frameSamples = np.array(findSignalEdges(analogInData['vsync'],edgeType='falling',thresh=-5000,refractory=2))

rfPath = fileIO.getFile('Select rf mapping pkl file',fileType='*.hdf5')
rfData = h5py.File(rfPath,'r')
frameIntervals = rfData['frameIntervals'][:]
frameRate = round(1/np.median(frameIntervals))
stimStart = rfData['stimStartFrame'][:-1]
stimStart += frameSamples.size-(frameIntervals.size+1)
stimPos = rfData['trialGratingCenter'][:len(stimStart)]

print(str(frameIntervals.size+1)+' frames')
print(str(frameSamples.size)+' frame samples')    


# LFP
preSamples = int(0.1*sampleRate)
postSamples = int(0.6*sampleRate)
azi,ele = [np.unique(p) for p in stimPos.T]
rfMapLFP = np.zeros((ele.size,azi.size,preSamples+postSamples))
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        trials = (stimPos[:,1]==y) & (stimPos[:,0]==x)
        startSamples = frameSamples[stimStart[trials]]
        for n,s in enumerate(startSamples):
            rfMapLFP[i,j] += np.mean(datData[channelRange[0]:channelRange[1]+1,s-preSamples:s+postSamples],axis=0)
        rfMapLFP[i,j] /= n+1

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(ele.size,azi.size)
ymin = 1.02*rfMapLFP.min()
ymax = 1.02*rfMapLFP.max()
t = np.arange(-preSamples,postSamples)/sampleRate
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        ax = fig.add_subplot(gs[ele.size-1-i,j])
        ax.plot(t,rfMapLFP[i,j],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        if i==0 and j==1:
            ax.set_xlabel('Time from stim onset (s)')
        else:
            ax.set_xticklabels([])
        if i==1 and j==0:
            ax.set_ylabel('LFP')
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,0.5])
        ax.set_ylim([ymin,ymax])
plt.tight_layout()


# spikes
negThresh = -200
posThresh = 50
chunkSamples = int(15*sampleRate)
subtractMedian = False

Wn = 300/(sampleRate/2) # cutoff freq normalized to nyquist
b,a = scipy.signal.butter(2,Wn,btype='highpass')

t = time.perf_counter()
spikes = []
offset = 0
while offset < datData.shape[1]:
    if offset > 0 and len(spikes) < 1:
        break
    d = datData[channelRange[0]:channelRange[1]+1,offset:offset+chunkSamples]
    if subtractMedian:
        d = d-np.median(d,axis=1)[:,None]
        d -= np.median(d,axis=0)
    d = scipy.signal.filtfilt(b,a,d,axis=1)
    for ch in d:
        s = np.array(findSpikes(ch,negThresh,posThresh)) + offset
        spikes.extend(s)
    offset += chunkSamples
print(time.perf_counter()-t)      

spikeTimes = np.array(spikes)/sampleRate
binSize = 1/frameRate
preTime = 0.1
postTime = 0.6
nbins = np.arange(0,preTime+postTime+binSize,binSize).size-1
azi,ele = [np.unique(p) for p in stimPos.T]
rfMapSpikes = np.zeros((ele.size,azi.size,nbins))
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        trials = (stimPos[:,1]==y) & (stimPos[:,0]==x)
        startTimes = frameSamples[stimStart[trials]]/sampleRate
        p,t = getPSTH(spikeTimes,startTimes-preTime,preTime+postTime,binSize=binSize,avg=True)
        rfMapSpikes[i,j] = p
t -= preTime

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(ele.size,azi.size)
ymax = 1.02*rfMapSpikes.max()
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        ax = fig.add_subplot(gs[ele.size-1-i,j])
        ax.plot(t,rfMapSpikes[i,j],'k')
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
        ax.set_ylim([0,ymax])
plt.tight_layout()

