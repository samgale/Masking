# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:19:26 2020

@author: svc_ccg
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import fileIO


f = fileIO.getFile()
d = h5py.File(f)

response = d['trialResponse'][:] # 1 for correct movement (or no movement on no-go trial), -1 for incorrect movement, 0 for no movement on go trial
nTrials = len(response)
targetFrames = d['trialTargetFrames'][:nTrials] # >0 if target presented
maskFrames = d['trialMaskFrames'][:nTrials] # >0 if mask presented
rewardDir = d['trialRewardDir'][:nTrials] # -1, 0, 1 for go left, no-go, go right

deltaWheel = d['deltaWheelPos'][:]
stimStart = d['trialStimStartFrame'][:nTrials]
openLoop = d['trialOpenLoopFrames'][:nTrials]
closedLoopStart = stimStart+openLoop+1
maxRespWait = d['maxResponseWaitFrames'][()]
wheelPos = np.cumsum(np.stack([deltaWheel[s:s+maxRespWait] for s in closedLoopStart]),axis=1)

normRewardDistance = d['normRewardDistance'][()]
monWidthPix = d['monSizePix'][0]
rewardThreshPix = normRewardDistance*monWidthPix
moveDir = np.zeros(nTrials)
for i,w in enumerate(wheelPos):
    move = np.where(np.absolute(w)>rewardThreshPix)[0]
    if len(move)>0:
        moveDir[i] = 1 if w[move[0]]>0 else -1
        
assert(np.all(moveDir[(rewardDir==1) & (response==1)]==1)) # correct right
assert(np.all(moveDir[(rewardDir==1) & (response==-1)]==-1)) # incorrect left
assert(np.all(moveDir[(rewardDir==-1) & (response==1)]==-1)) # correct left
assert(np.all(moveDir[(rewardDir==-1) & (response==-1)]==1)) # incorrect right




inputData = {}
inputData['target left, no mask'] = ((rewardDir==1) & (maskFrames>0))[1:nTrials]
inputData['target right, no mask'] = ((rewardDir==-1) & (maskFrames>0))[1:nTrials]
inputData['no target, no mask'] = ((rewardDir==0) & (maskFrames>0))[1:nTrials]
inputData['target left, mask'] = ((rewardDir==1) & (maskFrames==0))[1:nTrials]
inputData['target right, mask'] = ((rewardDir==-1) & (maskFrames==0))[1:nTrials]
inputData['no target, mask'] = ((rewardDir==0) & (maskFrames==0))[1:nTrials]
inputData['prev target left, move right (correct)'] = ((rewardDir==1) & (moveDir==1))[:nTrials-1]
inputData['prev target left, move left'] = ((rewardDir==1) & (moveDir==-1))[:nTrials-1]
inputData['prev target left no move'] = ((rewardDir==1) & (moveDir==0))[:nTrials-1]
inputData['prev target right, move left (correct)'] = ((rewardDir==-1) & (moveDir==-1))[:nTrials-1]
inputData['prev target right, move right'] = ((rewardDir==-1) & (moveDir==1))[:nTrials-1]
inputData['prev target right, no move'] = ((rewardDir==-1) & (moveDir==0))[:nTrials-1]
inputData['prev no target, no move (correct)'] = ((rewardDir==0) & (moveDir==0))[:nTrials-1]
inputData['prev no target, move right'] = ((rewardDir==0) & (moveDir==1))[:nTrials-1]
inputData['prev no target, move left'] = ((rewardDir==0) & (moveDir==-1))[:nTrials-1]




X = np.stack([inputData[key] for key in inputData]).T
y = moveDir[1:nTrials]


def plotConfusionMatrix(xdata,ydata,xlabel,ylabel):
    xvals = np.unique(xdata)
    yvals = np.unique(ydata)
    confusionMatrix = np.zeros((yvals.size,xvals.size))
    for i,y in enumerate(yvals):
        for j,x in enumerate(xvals):
            confusionMatrix[i,j] = np.sum(ydata[xdata==x]==y)/np.sum(xdata==x)
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(confusionMatrix,clim=(0,1),cmap='magma')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,round(confusionMatrix[i,j],2),color='w',horizontalalignment='center',verticalalignment='center')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['left','no','right'])
    ax.set_xlabel(xlabel)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['left','no','right'])
    ax.set_ylabel(ylabel)
    ax.set_xlim([-0.5,xvals.size-0.5])
    ax.set_ylim([yvals.size-0.5,-0.5])
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks([0,0.5,1])
    ax.set_title(ylabel+' probability given '+xlabel+
                 '\n'+'(overall accuracy = '+str(np.round(np.sum(ydata==xdata)/len(xdata),2))+')')
    plt.tight_layout()



trials = np.ones(y.size,dtype=bool) # inputData['mask']

plotConfusionMatrix(rewardDir,moveDir,'prescribed response','actual response')

# linear SVM
model = LinearSVC(C=1.0,max_iter=1e4)
model.fit(X[trials],y[trials])
accuracy = model.score(X,y)

plotConfusionMatrix(y,model.predict(X),'actual response','predicted response')

coef = model.coef_

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = np.max(np.absolute(coef))
im = ax.imshow(coef.T,clim=(-cmax,cmax),cmap='bwr')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['left','no','right'])
ax.set_xlabel('Response')
ax.set_yticks(np.arange(len(inputData)))
ax.set_yticklabels(inputData.keys())
ax.set_ylabel('Condition')
ax.set_xlim([-0.5,2.5])
ax.set_ylim([len(inputData)-0.5,-0.5])
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_title('Coefficient')
plt.tight_layout()


# random forest
model = RandomForestClassifier(n_estimators=100,oob_score=True)
model.fit(X[trials],y[trials])
accuracy = model.score(X,y)
oobAccuracy = model.oob_score_

plotConfusionMatrix(y,model.predict(X),'actual response','predicted response')








