# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import random
import numpy as np
import matplotlib.pyplot as plt


trialEnd = 650

Linitial = 0
Rinitial = 0
sigma = 0.5


threshold = 20


Lrecord = np.full(trialEnd,np.nan)
Rrecord = np.full(trialEnd,np.nan)

signalLatency = 40
Lsignal = np.zeros(trialEnd)
Rsignal = np.zeros(trialEnd)

Lsignal[signalLatency:signalLatency+17] = np.linspace(1,0,17)

L = Linitial
R = Rinitial

response = 0

for i in range(trialEnd):
    L += Lsignal[i] + random.gauss(0,sigma)
    R += Rsignal[i] + random.gauss(0,sigma)
    
    Lrecord[i] = L
    Rrecord[i] = R

    if L > threshold and R > threshold:
        response = -1 if L > R else 1
        break
    elif L > threshold:
        response = -1
        break
    elif R > threshold:
        response = 1
        break

responseTime = i+1



plt.figure()
plt.plot([0,trialEnd],[threshold,threshold],'k--')
plt.plot([0,trialEnd],[-threshold,-threshold],'k--')
plt.plot(Lrecord,'b')
plt.plot(-Rrecord,'r')





















