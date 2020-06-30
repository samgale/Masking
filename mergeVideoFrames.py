# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:57:18 2019

@author: svc_ccg
"""

from __future__ import division
import fileIO
import os
import h5py
import numpy as np
import skvideo
skvideo.setFFmpegPath('C:\\Users\\svc_ccg\\Desktop\\ffmpeg\\bin')
import skvideo.io


# get sync, behav cam, and screen cam data files
syncPath = fileIO.getFile('Select sync file',fileType='*.hdf5')
syncFile = h5py.File(syncPath,'r')

dirPath = os.path.dirname(syncPath)

behavCamPath = fileIO.getFile('Select behavior cam data',rootDir=dirPath,fileType='*.hdf5')
behavCamFile = h5py.File(behavCamPath,'r')

screenCamPath = fileIO.getFile('Select screen cam data',rootDir=dirPath,fileType='*.hdf5')
screenCamFile = h5py.File(screenCamPath,'r')


# get analog sync data
syncData = syncFile['AnalogInput']
syncSampleRate = syncData.attrs.get('sampleRate')
channelNames = syncData.attrs.get('channelNames')
behavCamSync = syncData[:,channelNames=='cam1Saving'][:,0]
screenCamSync = syncData[:,channelNames=='cam2Saving'][:,0]


# get rising times of camera frame signals
syncSampInt = 1/syncSampleRate
syncTime = np.arange(syncSampInt,syncSampInt*syncData.shape[0]+syncSampInt,syncSampInt)

frameTimes = []
for sync in (behavCamSync,screenCamSync):
    risingEdges = np.concatenate(([False],(sync[1:]-sync[:-1])>0.5))
    frameTimes.append(syncTime[risingEdges])
    frameIntervals = np.diff(frameTimes[-1])
    frameTimes[-1] = frameTimes[-1][np.concatenate(([True],frameIntervals>0.1*frameIntervals.mean()))]
behavCamFrameTimes,screenCamFrameTimes = frameTimes

assert(behavCamFrameTimes.size==behavCamFile['frames'].shape[0])
assert(screenCamFrameTimes.size==screenCamFile['frames'].shape[0])


# align screen cam to behavior cam for chosen screen cam frame range
screenCamFrameRange = (1,600)
screenCamFramesToShow = np.arange(screenCamFrameRange[0]-1,screenCamFrameRange[1])
screenCamTimesToShow = screenCamFrameTimes[screenCamFramesToShow]
behavCamFramesToShow = np.where((behavCamFrameTimes>=screenCamTimesToShow[0]) & (behavCamFrameTimes<=screenCamTimesToShow[-1]))[0]
behavCamTimesToShow = behavCamFrameTimes[behavCamFramesToShow]
alignedScreenCamFrames = screenCamFramesToShow[np.searchsorted(screenCamTimesToShow,behavCamTimesToShow)[:-1]]
alignedBehavCamFrames = behavCamFramesToShow[:-1]


# calculate merged frame shape
h1,w1 = behavCamFile['frames'].shape[1:]
h2,w2 = screenCamFile['frames'].shape[1:]
if h1>h2:
    offset1 = 0
    offset2 = int(0.5*(h1-h2))
else:
    offset1 = int(0.5*(h2-h1))
    offset2 = 0
gap = 2
mergedFrameShape = (h1+h2+gap,max(w1,w2))


# create merged video file
savePath = fileIO.saveFile('Save movie as',rootDir=dirPath,fileType='*.mp4')

inputParams = {'-r': str(behavCamFile.attrs.get('frameRate'))}
outputParams = {'-r': '30', '-vcodec': 'libx264', '-crf': '23', '-preset': 'slow'}

v = skvideo.io.FFmpegWriter(savePath,inputdict=inputParams,outputdict=outputParams)
mergedFrame = np.zeros(mergedFrameShape,dtype=np.uint8)
for i in range(alignedScreenCamFrames.size):
    mergedFrame[:h1,offset1:offset1+w1] = behavCamFile['frames'][alignedBehavCamFrames[i],:,:]
    mergedFrame[h1+gap:,offset2:offset2+w2] = screenCamFile['frames'][alignedScreenCamFrames[i],:,:]
    v.writeFrame(mergedFrame)
v.close()


# close hdf5 files
for f in (syncFile,behavCamFile,screenCamFile):
    f.close()
