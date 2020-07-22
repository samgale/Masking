# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import fileIO


@njit
def findRisingEdges(signal,thresh,refractory):
    """
    thresh: difference between current and previous value
    refractory: samples after detected edge to ignore
    """
    edges = []
    lastVal = signal[0]
    lastEdge = -refractory
    for i in range(1,signal.size):
        val = signal[i]
        if i-lastEdge>refractory and val-lastVal>thresh:
            lastEdge = i+1
            edges.append(lastEdge)
        lastVal = val
    return edges


# get analog sync data
syncPath = fileIO.getFile('Select sync file',fileType='*.hdf5')
syncFile = h5py.File(syncPath,'r')
syncData = syncFile['AnalogInput']
syncSampleRate = syncData.attrs.get('sampleRate')
channelNames = syncData.attrs.get('channelNames')
vsync = syncData[:,channelNames=='vsync'][:,0]
photodiode = syncData[:,channelNames=='photodiode'][:,0]
syncTime = np.arange(1/syncSampleRate,(syncData.shape[0]+1)/syncSampleRate,1/syncSampleRate)
syncFile.close()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(syncTime,vsync,'b')
ax.plot(syncTime,photodiode,'k')



# get probe data
probeDataDir = fileIO.getDir()

probeSampleRate = 30000

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
        units[u]['peakToTrough'] = np.argmax(template[peakInd:])/(probeSampleRate/1000)
    
    #check if this unit is noise
    tempNorm = template/np.max(np.absolute(template))
    units[u]['normTempIntegral'] = tempNorm.sum()
    if abs(tempNorm.sum())>4:
        units[u]['label'] = 'noise'


totalChannels = 136
probeChannels = 128      

rawData = np.memmap(os.path.join(probeDataDir,'continuous.dat'),dtype='int16',mode='r')    
rawData = np.reshape(rawData,(int(rawData.size/totalChannels),-1)).T


analogInData = {name: rawData[ch+probeChannels] for ch,name in enumerate(('vsync',
                                                                          'photodiode',
                                                                          'rotaryEncoder',
                                                                          'cam1Saving',
                                                                          'cam2Saving',
                                                                          'cam1Exposure',
                                                                          'cam2Exposure',
                                                                          'led'))}

    
frameSamples = findRisingEdges(analogInData['vsync'],thresh=15000,refractory=2)

behavDataPath = fileIO.getFile('',fileType='*.hdf5')
behavData = h5py.File(behavDataPath,'r')

assert(len(frameSamples)==behavData['frameIntervals'].size+1)










  