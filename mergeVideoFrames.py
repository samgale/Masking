# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:57:18 2019

@author: svc_ccg
"""

from __future__ import division
import fileIO
import os
import h5py
import cv2
import numpy as np


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
    risingEdges = np.concatenate(([False],(sync[1:]-sync[:-1])>0.1))
    frameTimes.append(syncTime[risingEdges])
    frameIntervals = np.diff(frameTimes[-1])
    frameTimes[-1] = frameTimes[-1][np.concatenate(([True],frameIntervals>0.1*frameIntervals.mean()))]
behavCamFrameTimes,screenCamFrameTimes = frameTimes

assert(behavCamFrameTimes.size==behavCamFile.attrs.get('numFrames'))
assert(screenCamFrameTimes.size==screenCamFile.attrs.get('numFrames'))


# align screen cam to behavior cam for chosen screen cam frame range
screenCamFrameRange = (87555,87897) 
screenCamFramesToShow = np.arange(screenCamFrameRange[0]-1,screenCamFrameRange[1])
screenCamTimesToShow = screenCamFrameTimes[screenCamFramesToShow]
behavCamFramesToShow = np.where((behavCamFrameTimes>=screenCamTimesToShow[0]) & (behavCamFrameTimes<=screenCamTimesToShow[-1]))[0]
behavCamTimesToShow = behavCamFrameTimes[behavCamFramesToShow]
alignedScreenCamFrames = screenCamFramesToShow[np.searchsorted(screenCamTimesToShow,behavCamTimesToShow)[:-1]]
alignedBehavCamFrames = behavCamFramesToShow[:-1]


# calculate merged frame shape
h1,w1 = behavCamFile['1'].shape
h2,w2 = screenCamFile['1'].shape
if h1>h2:
    offset1 = 0
    offset2 = int(0.5*(h1-h2))
else:
    offset1 = int(0.5*(h2-h1))
    offset2 = 0
gap = 2
mergedFrameShape = (h1+h2+gap,max(w1,w2))


# create merged video file
savePath = fileIO.saveFile(rootDir=dirPath)
mergedVideoFrameRate = behavCamFile.attrs.get('frameRate')

v = cv2.VideoWriter(savePath,-1,mergedVideoFrameRate,mergedFrameShape[::-1])
mergedFrame = np.zeros(mergedFrameShape,dtype=np.uint8)
for i in range(alignedScreenCamFrames.size):
    mergedFrame[:h1,offset1:offset1+w1] = behavCamFile[str(alignedBehavCamFrames[i]+1)][:,:]
    mergedFrame[h1+gap:,offset2:offset2+w2] = screenCamFile[str(alignedScreenCamFrames[i]+1)][:,:]
    v.write(mergedFrame)
v.release()


# close hdf5 files
for f in (syncFile,behavCamFile,screenCamFile):
    f.close()
