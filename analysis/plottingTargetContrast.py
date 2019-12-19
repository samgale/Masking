# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:57:58 2019

@author: chelsea.strawder
"""

import fileIO
import h5py, os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from behaviorAnalysis import formatFigure

matplotlib.rcParams['pdf.fonttype'] = 42

f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
d = h5py.File(f)

trialResponse = d['trialResponse'][:]
trialRewardDirection = d['trialRewardDir'][:len(trialResponse)]    # leave off last trial, ended session before answer
trialTargetFrames = d['trialTargetFrames'][:len(trialResponse)]
trialTargetContrast = d['trialTargetContrast'][:len(trialResponse)]
targetContrast = d['targetContrast'][:]

repeats = d['incorrectTrialRepeats'][()]


if 'trialRepeat' in d.keys():
    prevTrialIncorrect = d['trialRepeat'][:len(trialResponse)]  #recommended, since keeps track of how many repeats occurred 
else:
    prevTrialIncorrect = np.concatenate(([False],trialResponse[:-1]<1))     # array of boolean values, if trial before was incorr
trialResponse2 = trialResponse[prevTrialIncorrect==False]                   # false = not a repeat, true = repeat
trialRewardDirection = trialRewardDirection[prevTrialIncorrect==False]      # use this to filter out repeated trials 
trialTargetContrast = trialTargetContrast[prevTrialIncorrect==False]
trialTargetFrames = trialTargetFrames[prevTrialIncorrect==False]


# [R stim] , [L stim]
hits = [[],[]]
misses = [[], []]
noResps = [[],[]]

for i, direction in enumerate([-1,1]):
    directionResponses = [trialResponse2[
            (trialRewardDirection==direction) & (trialTargetContrast == tc)] for tc in np.unique(targetContrast)]
    hits[i].append([np.sum(drs==1) for drs in directionResponses])
    misses[i].append([np.sum(drs==-1) for drs in directionResponses])
    noResps[i].append([np.sum(drs==0) for drs in directionResponses])

hits = np.squeeze(np.array(hits))
misses = np.squeeze(np.array(misses))
noResps = np.squeeze(np.array(noResps))
totalTrials = hits+misses+noResps

# here call no_go movement function? 

if 0 in trialTargetFrames:        # this already excludes repeats 

    nogoTotal = len(trialTargetFrames[trialTargetFrames==0])
    nogoCorrect = len(trialResponse2[(trialResponse2==1) & (trialTargetFrames==0)])
    nogoMove = nogoTotal - nogoCorrect
    
    nogoTurnDir = []
  
    stimStart = d['trialStimStartFrame'][:][prevTrialIncorrect==False]
    trialOpenLoop = d['trialOpenLoopFrames'][:len(trialResponse)][prevTrialIncorrect==False]
    trialRespFrames = d['trialResponseFrame'][:][prevTrialIncorrect==False]   #gives the frame number of a response
    deltaWheel = d['deltaWheelPos'][:]
    
    stimStart = stimStart[(trialTargetFrames==0)]
    trialRespFrames = trialRespFrames[(trialTargetFrames==0)]
    trialOpenLoop = trialOpenLoop[(trialTargetFrames==0)]
    nogoResp = trialResponse2[(trialTargetFrames==0)]
    
    stimStart += trialOpenLoop
    
    startWheelPos = []
    endWheelPos = []
    
    # we want to see which direction they moved the wheel on an incorrect no-go
    for (start, end, resp) in zip(stimStart, trialRespFrames, nogoResp):   
        if resp==-1:
            endWheelPos.append(deltaWheel[end])
            startWheelPos.append(deltaWheel[start])
        
    endWheelPos = np.array(endWheelPos)
    startWheelPos = np.array(startWheelPos)   
    wheelPos = endWheelPos - startWheelPos
    
    for i in wheelPos:
        if i >0:
            nogoTurnDir.append(1)
        else:
            nogoTurnDir.append(-1)
    
    nogoTurnDir = np.array(nogoTurnDir)
    
    nogoR = sum(nogoTurnDir[nogoTurnDir==1])
    nogoL = sum(nogoTurnDir[nogoTurnDir==-1])*-1
else:
    pass
#misses = np.insert(misses, 0, [no_goR, no_goL], axis=1)  #add the no_go move trials to misses array 


for num, denom, title in zip([hits, hits, hits+misses], 
                             [totalTrials, hits+misses, totalTrials],
                             ['Percent Correct', 'Percent Correct Given Response', 'Total response rate']):
    fig, ax = plt.subplots()
    ax.plot(np.unique(targetContrast), num[0]/denom[0], 'bo-')  #here [0] is right trials and [1] is left
    ax.plot(np.unique(targetContrast), num[1]/denom[1], 'ro-')
   # ax.plot(np.unique(targetContrast), (num[0]+num[1])/(denom[0]+denom[1]), 'ko--', alpha=.3)  #plots the combined average 
    y1=(num[0]/denom[0])
    y2=(num[1]/denom[1])
    for i, length in enumerate(np.unique(targetContrast)):
        plt.annotate(str(denom[0][i]), xy=(length,y1[i]), xytext=(5, -10), textcoords='offset points')  
        plt.annotate(str(denom[1][i]), xy=(length,y2[i]), xytext=(-10, 10), textcoords='offset points')
    
    if 0 in trialTargetFrames:
        ax.plot(0, nogoCorrect/nogoTotal, 'go') 
        if title=='Total response rate':
            ax.plot(0, nogoR/nogoMove, 'g>')   #plot the side that was turned in no-go with an arrow in that direction
            ax.plot(0, nogoL/nogoMove, 'g<')  #add counts
       
    formatFigure(fig, ax, xLabel='Target Contrast', yLabel='percent trials', 
                 title=title + " :  " + '-'.join(f.split('_')[-3:-1]))
    ax.set_xlim([-.1, targetContrast[-1]+.1])
    ax.set_ylim([0,1.05])
    ax.set_xticks(np.unique(trialTargetContrast))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)

    #ax.text(np.unique(targetContrast), (num[0]/denom[0]), str(denom))
            
    if 0 in trialTargetFrames:   
        a = ax.get_xticks().tolist()
        #a = [int(i) for i in a]    
        a[0]='no-go' 
        ax.set_xticklabels(a)