# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:40:50 2023

@author: svc_ccg
"""

import glob
import os
import h5py
import numpy as np
import cv2
import skvideo
skvideo.setFFmpegPath('C:\\Users\\svc_ccg\\Desktop\\ffmpeg\\bin')
import skvideo.io
from maskTaskAnalysisUtils import MaskTaskData


outputDir = r"\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Sam\Presentations\video_example"

behavFilesDir = r"\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Sam\Analysis\Data\masking\wt"

dataDir = r"\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Data"

stimNames = ('target_left','target_right','mask','gray')
leftTarget,rightTarget,mask,gray = [cv2.imread(os.path.join(outputDir,stim+'.png'),cv2.IMREAD_GRAYSCALE) for stim in stimNames]


behavFiles = glob.glob(os.path.join(behavFilesDir,'*.hdf5'))
for f in behavFiles:
    obj = MaskTaskData()
    obj.loadBehavData(f)
    
    mouseId = f[-27:-21]
    expDate = f[-16:-12] + f[-20:-16]
    
    syncPath = glob.glob(os.path.join(dataDir,mouseId+'_'+expDate,'Sync*.hdf5'))
    if len(syncPath) == 0:
        continue
    syncPath = syncPath[0]
    videoPath = glob.glob(os.path.join(dataDir,mouseId+'_'+expDate,'BehavCam*.mp4'))[0]
    
    syncFile = h5py.File(syncPath,'r')
    syncData = syncFile['AnalogInput']
    syncSampleRate = syncData.attrs.get('sampleRate')
    syncChannelNames = syncData.attrs.get('channelNames')
    
    syncSampInt = 1/syncSampleRate
    syncTime = np.arange(syncSampInt,syncSampInt*syncData.shape[0]+syncSampInt,syncSampInt)
    
    vsync = syncData[:,np.where(syncChannelNames=='vsync')[0][0]]
    fallingEdges = np.concatenate(([False],(vsync[1:]-vsync[:-1]) < -0.5))
    vsyncTimes = syncTime[fallingEdges]
    vsyncIntervals = np.diff(vsyncTimes)
    vsyncTimes = vsyncTimes[np.concatenate(([True],vsyncIntervals>0.1*vsyncIntervals.mean()))]
    
    if vsyncTimes.size != obj.behavFrameIntervals.size+1:
        continue
    
    camSync = syncData[:,np.where(syncChannelNames=='cam1Exposure')[0][0]]
    risingEdges = np.concatenate(([False],(camSync[1:]-camSync[:-1]) > 0.5))
    camSyncTimes = syncTime[risingEdges]
    camSyncIntervals = np.diff(camSyncTimes)
    camSyncTimes = camSyncTimes[np.concatenate(([True],camSyncIntervals>0.1*camSyncIntervals.mean()))]
    
    camFrameData = h5py.File(os.path.join(videoPath[:-3]+'hdf5'),'r')
    camFrameTimes = camFrameData['frameTimes'][()]
    camFrameTimes -= camFrameTimes[0]
    camFrameIntervals = np.diff(camFrameTimes)
    
    camFrameIndex = np.searchsorted(camSyncTimes,vsyncTimes)
    
    
    videoIn = cv2.VideoCapture(videoPath)
    isFrame,videoFrame = videoIn.read()
    videoFrame = cv2.cvtColor(videoFrame,cv2.COLOR_BGR2GRAY)
    
    
    exampleTrials = []
    for i in range(obj.ntrials-3):
        if np.all((obj.trialType[i:i+3] == ('targetOnly','mask','mask')) & 
                  (obj.maskOnset[i:i+3] == (0,6,2)) &
                  (obj.response[i:i+3] == (1,1,1))):
            exampleTrials.append(i)
    
    
    inputParams = {'-r': '60'}
    outputParams = {'-r': '60', '-vcodec': 'libx264', '-crf': '23', '-preset': 'slow'}
    preFrames = 60
    postFrames = 120
    crop = np.s_[50:,50:320]
    offsetX = int(0.5*(videoFrame[crop].shape[1] - mask.shape[1]))
    offsetY = mask.shape[0]
    for trialIndex in exampleTrials:
        videoData = np.zeros((3*(preFrames+postFrames),videoFrame[crop].shape[0]+mask.shape[0],videoFrame[crop].shape[1]),dtype=np.uint8)
        videoData[:,:mask.shape[0],offsetX:offsetX+mask.shape[1]] = gray[None,:,:]
        for i in range(3):
            j = i*(preFrames+postFrames)+preFrames
            videoData[j,:mask.shape[0],offsetX:offsetX+mask.shape[1]] = leftTarget if obj.rewardDir[trialIndex+i]==1 else rightTarget
            maskOnset = int(obj.maskOnset[trialIndex+i]/2)
            if maskOnset > 0:
                videoData[j+maskOnset:j+39,:mask.shape[0],offsetX:offsetX+mask.shape[1]] = mask[None,:,:]
            
            startFrame = obj.trialStartFrame[trialIndex+i]
            for j,behavFrame in enumerate(range(startFrame-preFrames*2,startFrame+postFrames*2,2)):
                videoIn.set(cv2.CAP_PROP_POS_FRAMES,camFrameIndex[behavFrame])
                isFrame,videoFrame = videoIn.read()
                videoFrame = cv2.cvtColor(videoFrame,cv2.COLOR_BGR2GRAY)
                videoData[i*(preFrames+postFrames)+j,offsetY:,:] = videoFrame[crop]
        
        savePath = os.path.join(outputDir,'masking_example_'+mouseId+'_'+expDate+'_trial'+str(trialIndex)+'.mp4')
        v = skvideo.io.FFmpegWriter(savePath,inputdict=inputParams,outputdict=outputParams)
        for d in videoData:
            v.writeFrame(d)
        v.close()


