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


stimNames = ('left_target','right_target','mask')

d = []
for stim in stimNames:
    img = cv2.imread(os.path.join(outputDir,stim+'.png'),cv2.IMREAD_UNCHANGED)
    if len(img.shape)>2:
        img = img[:,:,::-1]
    d.append(img)
    
leftTarget,rightTarget,mask = d




behavFiles = glob.glob(os.path.join(behavFilesDir,'*.hdf5'))

f = behavFiles[-3]


obj = MaskTaskData()
obj.loadBehavData(f)

mouseId = f[-27:-21]
expDate = f[-16:-12] + f[-20:-16]

syncPath = glob.glob(os.path.join(dataDir,mouseId+'_'+expDate,'Sync*.hdf5'))[0]
                     
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

assert(vsyncTimes.size==obj.behavFrameIntervals.size+1)


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


exampleTrials = []
for i in range(obj.ntrials-3):
    if np.all((obj.trialType[i:i+3] == ('targetOnly','mask','mask')) & 
              (obj.maskOnset[i:i+3] == (0,6,2)) &
              (obj.response[i:i+3] == (1,1,1))):
        exampleTrials.append(i)

        
trial = exampleTrials[0]

preFrames = 60
postFrames = 120
for i in range(trial,trial+3):
    startFrame = obj.trialStartFrame[i]
    frameIndex = np.arange(startFrame-preFrames,startFrame+postFrames)
    
    obj.rewardDir[i]
    obj.maskOnset[i]
    



isImage,image = self.videoIn.read()
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

videoIn.set(cv2.CAP_PROP_POS_FRAMES,0)



savePath = os.path.join(outputDir,'masking_example.mp4')

inputParams = {'-r': str(behavCamFile.attrs.get('frameRate'))}
outputParams = {'-r': '30', '-vcodec': 'libx264', '-crf': '23', '-preset': 'slow'}

# '-pix_fmt': 'yuv420p' number of pixels needs to be even

v = skvideo.io.FFmpegWriter(savePath,inputdict=inputParams,outputdict=outputParams)
mergedFrame = np.zeros(mergedFrameShape,dtype=np.uint8)
for i in range(alignedScreenCamFrames.size):
    mergedFrame[:h1,offset1:offset1+w1] = behavCamFile['frames'][alignedBehavCamFrames[i],:,:]
    mergedFrame[h1+gap:,offset2:offset2+w2] = screenCamFile['frames'][alignedScreenCamFrames[i],:,:]
    v.writeFrame(mergedFrame)
v.close()


