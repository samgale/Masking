# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

from __future__ import division
import fileIO
import h5py
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


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



# get data
f = fileIO.getFile(rootDir=r'C:/Users/SVC_CCG/Desktop/Data/')

d = h5py.File(f)

frameRate = int(round(d['frameRate'].value))

preStimFrames = d['preStimFrames'].value

trialStartFrame = d['trialStartFrame'][:]

trialEndFrame = d['trialEndFrame'][:]

trialResponse = d['trialResponse'][:]

trialMaskOnset = d['trialMaskOnset'][:trialResponse.size]

encoderAngle = d['rotaryEncoderRadians'][:]


# calculate reaction time
angleChange = np.concatenate(([0],np.diff(encoderAngle)))
angleChange[angleChange<-np.pi] += 2*np.pi
angleChange[angleChange>np.pi] -= 2*np.pi
angleChange = scipy.signal.medfilt(angleChange,5)
reactionThresh = 0.1
reactionTime = np.full(trialResponse.size,np.nan)
for trial,(start,end) in enumerate(zip(trialStartFrame,trialEndFrame)):
    r = np.where(np.absolute(angleChange[start+preStimFrames:end])>reactionThresh)[0]
    if any(r):
        reactionTime[trial] = r[0]/frameRate


# determine fraction correct for each mask onset delay
trialMaskOnset[np.isnan(trialMaskOnset)] = 2*np.nanmax(trialMaskOnset)
maskOnsets = np.unique(trialMaskOnset)
numTrials = np.zeros(maskOnsets.size)
numIncorrect = numTrials.copy()
numNoResp = numTrials.copy()
numCorrect = numTrials.copy()
reactionTimeCorrect = numTrials.copy()
reactionTimeIncorrect = numTrials.copy()
outcomeTimeCorrect = numTrials.copy()
outcomeTimeIncorrect = numTrials.copy()
for i,mo in enumerate(maskOnsets):
    moTrials = trialMaskOnset==mo
    incorrect,noResp,correct = [trialResponse==j for j in (-1,0,1)]
    numTrials[i] = moTrials.sum()
    numIncorrect[i],numNoResp[i],numCorrect[i] = [np.sum(moTrials & trials) for trials in (incorrect,noResp,correct)]
    reactionTimeIncorrect[i],reactionTimeCorrect[i] = [np.nanmean(reactionTime[moTrials & trials]) for trials in (incorrect,correct)]
    outcomeTimeIncorrect[i],outcomeTimeCorrect[i] = [(np.mean(trialEndFrame[moTrials & trials])-preStimFrames)/frameRate for trials in (incorrect,correct)]
fractionCorrect = (numCorrect/(numCorrect+numIncorrect))
    
    
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot([0,maskOnsets[-1]],[0.5,0.5],'--',color='0.5')
ax.plot(maskOnsets/frameRate,fractionCorrect,'ko-',ms=10)
for side in ('top','right'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(np.array([-0.02,1.02])*maskOnsets[-1]/frameRate)
ax.set_ylim([0,1.02])
ax.set_xlabel('Stimulus onset asynchrony (s)',fontsize=16)
ax.set_ylabel('Fraction Correct',fontsize=16)
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot(maskOnsets/frameRate,reactionTimeIncorrect,'bo',ms=10)
ax.plot(maskOnsets/frameRate,reactionTimeCorrect,'ro',ms=10)
for side in ('top','right'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(np.array([-0.02,1.02])*maskOnsets[-1]/frameRate)
ax.set_xlabel('Stimulus onset asynchrony (s)',fontsize=16)
ax.set_ylabel('Reaction time (s)',fontsize=16)
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot(maskOnsets/frameRate,outcomeTimeIncorrect,'bo',ms=10)
ax.plot(maskOnsets/frameRate,outcomeTimeCorrect,'ro',ms=10)
for side in ('top','right'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim(np.array([-0.02,1.02])*maskOnsets[-1]/frameRate)
ax.set_xlabel('Stimulus onset asynchrony (s)',fontsize=16)
ax.set_ylabel('Outcome time (s)',fontsize=16)
plt.tight_layout()


  