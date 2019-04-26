# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:57:18 2019

@author: svc_ccg
"""

import fileIO
import h5py
import cv2
import numpy as np

syncFilePath = fileIO.getFile('*.hdf5')
dataFile = h5py.File(openFilePath)

saveFilePath = openFilePath[:-4]
saveFilePath += 'avi'

frameRate = dataFile.attrs.get('frameRate')
numFrames = dataFile.attrs.get('numFrames')

h1,w1 = dataFile['1'].shape
h2,w2 = dataFile['w1'].shape

if w1>w2:
    offset1 = 0
    offset2 = int(0.5*(w1-w2))
else:
    offset1 = int(0.5*(w2-w1))
    offset2 = 0
    
gap = 2

frameShape = (h1+h2+gap,max(w1,w2))

v = cv2.VideoWriter(saveFilePath,-1,frameRate,frameShape[::-1])

for frame in range(1,numFrames+1):
    d = np.zeros(frameShape,dtype=np.uint8)
    d[:h1,offset1:offset1+w1] = dataFile[str(frame)][:,:]
    d[h1+gap:,offset2:offset2+w2] = dataFile['w'+str(frame)][:,:]
    v.write(d)

v.release()
dataFile.close()
