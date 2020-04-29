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
closedLoopStart = stimStart+openLoop
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



def plotConfusionMatrix(xdata,ydata,xlabel,ylabel,ticklabels):
    xvals = np.unique(xdata)
    yvals = np.unique(ydata)
    confusionMatrix = np.zeros((yvals.size,xvals.size))
    for i,y in enumerate(yvals):
        for j,x in enumerate(xvals):
            confusionMatrix[i,j] = np.sum(ydata[xdata==x]==y)/np.sum(xdata==x)
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(confusionMatrix,clim=(0,1),cmap='magma')
    for i in range(len(yvals)):
        for j in range(len(xvals)):
            ax.text(j,i,round(confusionMatrix[i,j],2),color='w',horizontalalignment='center',verticalalignment='center')
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels(ticklabels,rotation=45,horizontalalignment='right')
    ax.set_xlabel(xlabel)
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels(ticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-0.5,xvals.size-0.5])
    ax.set_ylim([yvals.size-0.5,-0.5])
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks([0,0.5,1])
    ax.set_title(ylabel+' probability given '+xlabel+
                 '\n'+'(overall accuracy = '+str(np.round(np.sum(ydata==xdata)/len(xdata),2))+')')
    plt.tight_layout()



# simple
inputData = {}
inputData['target'] = rewardDir[1:nTrials]
inputData['mask'] = maskFrames[1:nTrials]>0
inputData['prev target'] = rewardDir[:nTrials-1]
inputData['prev mask'] = maskFrames[:nTrials-1]>0
inputData['prev response'] = moveDir[:nTrials-1]
inputData['prev reward'] = response[:nTrials-1]>0

outputData = {}
outputData['move left'] = moveDir[1:nTrials]==-1
outputData['no move'] = moveDir[1:nTrials]==0
outputData['move right'] = moveDir[1:nTrials]==1


# complex
inputData = {}
inputData['target left, no mask'] = ((rewardDir==1) & (maskFrames==0))[1:nTrials]
inputData['target right, no mask'] = ((rewardDir==-1) & (maskFrames==0))[1:nTrials]
inputData['no target, no mask'] = ((rewardDir==0) & (maskFrames==0))[1:nTrials]
inputData['target left, mask'] = ((rewardDir==1) & (maskFrames>0))[1:nTrials]
inputData['target right, mask'] = ((rewardDir==-1) & (maskFrames>0))[1:nTrials]
inputData['no target, mask'] = ((rewardDir==0) & (maskFrames>0))[1:nTrials]
inputData['prev target left, move right (correct)'] = ((rewardDir==1) & (moveDir==1))[:nTrials-1]
inputData['prev target left, move left'] = ((rewardDir==1) & (moveDir==-1))[:nTrials-1]
inputData['prev target left no move'] = ((rewardDir==1) & (moveDir==0))[:nTrials-1]
inputData['prev target right, move left (correct)'] = ((rewardDir==-1) & (moveDir==-1))[:nTrials-1]
inputData['prev target right, move right'] = ((rewardDir==-1) & (moveDir==1))[:nTrials-1]
inputData['prev target right, no move'] = ((rewardDir==-1) & (moveDir==0))[:nTrials-1]
inputData['prev no target, no move (correct)'] = ((rewardDir==0) & (moveDir==0))[:nTrials-1]
inputData['prev no target, move right'] = ((rewardDir==0) & (moveDir==1))[:nTrials-1]
inputData['prev no target, move left'] = ((rewardDir==0) & (moveDir==-1))[:nTrials-1]

outputData = {}
outputData['move left correct'] = ((rewardDir==-1) & (moveDir==-1))[1:nTrials]
outputData['no move correct'] = ((rewardDir==0) & (moveDir==0))[1:nTrials]
outputData['move right correct'] = ((rewardDir==1) & (moveDir==1))[1:nTrials]
outputData['move left incorrect'] = ((rewardDir!=-1) & (moveDir==-1))[1:nTrials]
outputData['no move incorrect'] = ((rewardDir!=0) & (moveDir==0))[1:nTrials]
outputData['move right incorrect'] = ((rewardDir!=1) & (moveDir==1))[1:nTrials]



X = np.stack([inputData[key] for key in inputData]).T
y = np.where(np.stack([outputData[key] for key in outputData]).T)[1]

trials = np.ones(y.size,dtype=bool) # inputData['mask']

plotConfusionMatrix(rewardDir,moveDir,'prescribed response','actual response',outputData.keys())



# linear SVM
model = LinearSVC(C=1.0,max_iter=1e4)
model.fit(X[trials],y[trials])
accuracy = model.score(X,y)
coef = model.coef_

plotConfusionMatrix(y,model.predict(X),'actual response','predicted response',outputData.keys())

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = np.max(np.absolute(coef))
im = ax.imshow(coef.T,clim=(-cmax,cmax),cmap='bwr')
xvals = np.unique(y)
ax.set_xticks(xvals)
ax.set_xticklabels(outputData.keys(),rotation=45,horizontalalignment='right')
ax.set_xlabel('Response')
ax.set_yticks(np.arange(len(inputData)))
ax.set_yticklabels(inputData.keys())
ax.set_ylabel('Feature')
ax.set_xlim([-0.5,len(outputData)-0.5])
ax.set_ylim([len(inputData)-0.5,-0.5])
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_title('Coefficient')
plt.tight_layout()


# random forest
model = RandomForestClassifier(n_estimators=100,oob_score=True)
model.fit(X[trials],y[trials])
accuracy = model.score(X,y)
oobAccuracy = model.oob_score_
featureImportance = model.feature_importances_

plotConfusionMatrix(y,model.predict(X),'actual response','predicted response',outputData.keys())

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xvals = np.arange(len(inputData))
ax.bar(xvals,featureImportance,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xvals)
ax.set_xticklabels(inputData.keys(),rotation=45,horizontalalignment='right')
ax.set_xlabel('Feature')
ax.set_ylabel('Feature importance')
ax.set_xlim([-0.5,len(inputData)-0.5])
ax.set_ylim([0,1.05*featureImportance.max()])
plt.tight_layout()








