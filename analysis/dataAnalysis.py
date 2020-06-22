# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:02:06 2020

@author: chelsea.strawder

My hope is to combine and condense all of the data analysis functions into a single file
that we can then use to analyze the session data in a simpler script 
use:
    d = import_data()
    df = create_df(d)
OR --> df = create_df(import_data())
"""

import fileIO, h5py
import numpy as np
import pandas as pd
from nogoData import nogo_turn
from collections import defaultdict


def import_data():
    f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
    d = h5py.File(f)
    return d



def combine_files(files, *dates, output='df'):
    ''' use output = 'd' if you just want the files returned, otherwise df is default
    write dates as mdd  ex: 212 for Feb 12 - might need to extend with years in future
    
    use with behaviorAnalysis.get_files() for files '''
    
    dict1 = {}
    filtered_files = [file for file in files for date in dates if date in file]
    for i, f in enumerate(filtered_files):  
        d = h5py.File(f) 
        if output=='df':
            dict1['df_{}'.format(i)] = create_df(d)
        else:
            dict1['df_{}'.format(i)] = d
    return dict1



def combine_dfs(dict1):
    ''' to be used with dict of dataframes
    retains metadata from original dfs'''
    
    df = pd.concat(dict1.values(), ignore_index=True)
    df.mouse = {val.mouse for val in dict1.values()}   #preserve metadata
    df.date = [val.date for val in dict1.values()]
    df.framerate = [val.framerate for val in dict1.values()]
    return df



def create_df(data):   
    
## PULL RELEVANT DATA FROM HDF5 FILE TO CREATE DATAFRAME 
    
    d = data
    mouse, date = str(d).split('_')[-3:-1]
        
    trialResponse = d['trialResponse'][:]
    end = len(trialResponse)
    trialRewardDirection = d['trialRewardDir'][:end]
    trialTargetFrames = d['trialTargetFrames'][:end]
    trialTargetContrast = d['trialTargetContrast'][:end]
    
    trialStartFrame = d['trialStartFrame'][:end]
    trialStimStartFrame = d['trialStimStartFrame'][:end]
    trialResponseFrame = d['trialResponseFrame'][:end] 
    trialEndFrame = d['trialEndFrame'][:end]
    postReward = d['postRewardTargetFrames'][()]
    
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))
    
    def convert_to_ms(value):
        return value * 1000/framerate

    trialOpenLoopFrames = d['trialOpenLoopFrames'][:end]
    if len(np.unique(trialOpenLoopFrames)>1):
        pass
        
    # vars below are to look at turning behavior when prestim frames are not predictable
    #    preStimFrames = d['preStimFramesFixed'][()]
    #    preStimVar = d['preStimFramesVariableMean'][()]              
    #    openLoopFrames = d['openLoopFramesFixed'][()]
    #    openLoopVar = d['openLoopFramesVariableMean'][()]
    #    openLoopMax = d['openLoopFramesMax'][()]
      
    quiescentMoveFrames = [q for q in d['quiescentMoveFrames'][:] if q<trialStimStartFrame[-1]]
    
    maxResp = d['maxResponseWaitFrames'][()]
    deltaWheelPos = d['deltaWheelPos'][:]   
                   
    repeats = d['trialRepeat'][:end]
    #nogoWait = d['nogoWaitFrames'][()]   # these are often different than target wait frames
        
    maskOnset = convert_to_ms(d['maskOnset'][()])
    trialMaskOnset = convert_to_ms(d['trialMaskOnset'][:end])
    trialMaskContrast = d['trialMaskContrast'][:end]

    
### PROCESS AND CLEAN DATA
    
    for i, target in enumerate(trialTargetFrames):  # this is needed for older files nogos are randomly assigned a dir
        if target==0 and np.isfinite(trialRewardDirection[i]):   #nan is reserved for catch trials
            trialRewardDirection[i] = 0
    
    nogos = [i for i, (rew, con) in enumerate(zip
             (trialRewardDirection, trialMaskContrast)) if rew==0 and con==0]
   
    if np.any(trialMaskOnset>0):
        targetOnlyVal = maskOnset[-1] + 0.5*maskOnset[-1] 
        #round(np.mean(np.diff(maskOnset)))  # assigns targetOnly condition an evenly-spaced value from soas
        maskOnset = np.append(maskOnset, targetOnlyVal)                     # makes final value the targetOnly condition
            
        for i, (mask, trial) in enumerate(zip(trialMaskOnset, trialTargetFrames)):   # filters target-Only trials 
            if trial>0 and mask==0:
                trialMaskOnset[i]=targetOnlyVal      
    
    trialLength = trialResponseFrame - trialStimStartFrame

        # gives entire wheel trace from trial start to the max trial length
        # i.e. often goes into the start of the next trial - 
        # useful for analyzing turning beh on nogo/mask only
    deltaWheel = [deltaWheelPos[start:stim+openLoop+maxResp+postReward] for (start, stim, openLoop) in 
                  zip(d['trialStartFrame'][:len(trialStimStartFrame)],
                      trialStimStartFrame, trialOpenLoopFrames)]
    
    turns, inds = nogo_turn(d)      #for both of these, [0]=nogos, [1]=maskOnly                    
    
        #frame intervals for each trial, start to end
    frames = [fi[start:end] for (start, end) in zip(trialStartFrame[:len(trialEndFrame)], trialEndFrame)]
    if len(trialEndFrame) < len(trialStartFrame):
        frames.append(fi[trialStartFrame[-1]:trialResponseFrame[-1]])
    
    trueMaskOnset = [sum(fi[stim:stim+mask]) for (stim, mask) in 
                     zip(trialStimStartFrame, d['trialMaskOnset'])]
    
    qDict = defaultdict(list)
    for i, (start,stimStart) in enumerate(zip(trialStartFrame, trialStimStartFrame)):
        for x in quiescentMoveFrames:    
            if start<x<stimStart:
                qDict[i].append(x)

    assert len(quiescentMoveFrames) == sum([len(qDict[x]) for x in qDict]), "Qframes Error"
                      
### CREATE DATAFRAME
                
    data = list(zip(trialRewardDirection, trialResponse, 
                    trialStartFrame, trialStimStartFrame, trialResponseFrame, trialOpenLoopFrames))

    index = np.arange(len(trialResponse))
    df = pd.DataFrame(data, 
                      index=index, 
                      columns=['rewDir', 'resp', 'trialStart', 'stimStart', 'respFrame', 'openLoopFrames'])
    
    df['trialLength_ms'] = convert_to_ms(trialLength)
    
    df['mask'] = trialMaskContrast
    df['soa'] = trialMaskOnset
    #df['maskLength_ms'] = convert_to_ms(d['trialMaskFrames'][:end])
    df['maskContrast'] = trialMaskContrast

    df['targetDuration_ms'] = convert_to_ms(trialTargetFrames)
    df['targetContrast'] = trialTargetContrast
    
    df['nogo'] = False
    for i in nogos:
        df.loc[i, 'nogo'] = True
        df.loc[i, 'soa'] = -maskOnset[0]  # this helps for summary stats
   
    def fill():
        return np.zeros(len(trialResponse)).astype(int)
    
    df['nogoMove'] = fill()
    df['maskOnlyMove'] = fill()
          
    for e, col in enumerate(('nogoMove', 'maskOnlyMove')):
        for (i,turn) in zip(inds[e], turns[e]):
            df.at[i, col] = turn
        
    df['repeat'] = repeats    
    
    df['quiescentViolations'] = fill()
    for key,val in qDict.items():
        df.at[key, 'quiescentViolations'] = len(val)
        
    df['deltaWheel'] = deltaWheel
    
    df['trialFrameIntervals'] = frames
 
    df['soa_frames'] = d['trialMaskOnset'][:len(df)]  
    df['actualSOA_ms'] = np.array(trueMaskOnset) * 1000
 
    df.mouse = mouse
    df.date = date
    df.framerate = framerate
    
        ## call reaction time function (defined below)
    times = rxnTimes(d, df)  # 0==initiation, 1==outcome, 2==ignore
    
    df['initiationTime_ms'] = times[0]
    df['outcomeTime_ms'] = times[1]
    df['ignoreTrial'] = False   
    for i in times[2]:
        df.loc[i, 'ignoreTrial'] = True
    
    return df



def get_dates(dataframe):
    
    from datetime import datetime
    df=dataframe
    if type(df.date) is list:
        dates = [datetime.strptime(date, '%Y%m%d').strftime('%m/%d/%Y') for date in df.date]
        date = '-'.join([dates[0], dates[-1]])
    else:
        date = datetime.strptime(df.date, '%Y%m%d').strftime('%m/%d/%Y')
        
    return date




def rxnTimes(data, dataframe):
    
    d = data
    df = dataframe
            
    monitorSize = d['monSizePix'][0] 
    maxResp = d['maxResponseWaitFrames'][()]
    
# these 2nd values are needed for files after 6/18/2020    
    normRewardDist = d['normRewardDistance'][()] if 'wheelRewardDistance' not in d.keys()\
    else d['wheelRewardDistance'][()]  #normalized to screen width in pixels- used for wheelgain
    maxQuiescentMove = d['maxQuiescentNormMoveDist'][()] if 'maxQuiescentNormMoveDist' in d.keys()\
    else d['maxQuiescentMoveDist'][()]   # in mm
    wheelSpeedGain = d['wheelSpeedGain'][()] if 'wheelSpeedGain' in d.keys() else d['wheelGain'][()]
    
## Here need to stop using screen and start using amount wheel turned 
    wheelRad = d['wheelRadius'][()]
    wheelRewardDist = d['wheelRewardDistance'][()]
    
    initiationThreshDeg = 0.5 
    initiationThreshPix = initiationThreshDeg*np.pi/180*wheelSpeedGain
    sigThreshold = maxQuiescentMove * monitorSize
    rewThreshold = normRewardDist * monitorSize

    fi = d['frameIntervals'][:]
    
    stimInds = df['stimStart'] - df['trialStart']

    interpWheel = []
    initiateMovement = []
    significantMovement = []
    ignoreTrials = []  # old ignore_trials func actually calls current func and returns this list
    outcomeTimes = []
    
    ## use below code to determine wheel direction changes during trial 
    # during just trial time (ie in nogos before trial ends) or over entire potential time? both?

 
    for i, (wheel, resp, rew, fiInd, stimInd, openLoop) in enumerate(zip(
            df['deltaWheel'],df['resp'], df['rewDir'], df['stimStart'], stimInds, df['openLoopFrames'])):

        
        f = fi[fiInd:(fiInd+(len(wheel)-stimInd))]   #from stim start frame to len of maxTrial + 24 frames 
        wheel *= wheelRad
        fp = np.cumsum(wheel[stimInd:stimInd+len(f)])   
        xp = np.cumsum(f)    
        x = np.arange(0, xp[-1], .001)
        interp = np.interp(x,xp,fp)
        interpWheel.append(interp)
      

        sigMove = np.argmax(abs(interp)>=sigThreshold)   # or just > ??
        significantMovement.append(sigMove)
        if 0<sigMove<150:
            ignoreTrials.append(i)
        
        
        if (rew==0) and (resp==1):
            init = 0 
        else:
            init = np.argmax(abs(interp)>initiationThreshPix)

        if (0<init<100) and sigMove>150:
            init = np.argmax(abs(interp[100:])>(initiationThreshPix + interp[100])) + 100
        # does this handle turning the opposite direction ?
        
        initiateMovement.append(init)
        
        # also want time from start of choice til choice  (using modified version of sam's method)
                 
        if rew>0:
             outcome = np.argmax(interp[150:] >= rewThreshold + interp[150]) + 150
        elif rew<0:  
            outcome = np.argmax(interp[150:] <= (rew*rewThreshold) + interp[150]) + 150
        else:
            outcome = np.argmax(abs(interp[150:]) >= (abs(rewThreshold) + interp[150])) + 150
  
        if outcome==150:
            outcomeTimes.append(0)
        else:
            outcomeTimes.append(outcome)

    return np.array([initiateMovement, outcomeTimes, ignoreTrials])


## code to plot the above wheel traces, to visually inspect for accuracy
#test = [i for i, e in enumerate(interpWheel) if type(e)!=int]

#catchTrials = [i for i, row in df.iterrows() if row.isnull().any()]
#
#plt.figure()
#for i in catchTrials:
#    
#    plt.plot(interpWheel[i], color='k', alpha=.5)
#    plt.suptitle('From Stim start to max trial len')
#    plt.title('Reward ' + df.loc[i, 'rewDir'].astype(str) + '  , Response ' + df.loc[i, 'resp'].astype(str))
#    
#plt.vlines(200, -1000, 1000, color='g', ls ='--', label='Closed Loop', lw=2)
##plt.vlines(initiateMovement[i], -400, 400, ls='--', color='m', alpha=.4, label='Initiation')
##plt.vlines(significantMovement[i], -400, 400, ls='--', color='c', alpha=.4 , label='Q threshold')
##plt.vlines(outcomeTimes[i], -400, 400, ls='--', color='b', alpha=.3, label='Outcome Time')
##plt.vlines(df['trialLength_ms'][i], -500, 500, label='Trial Length')
#plt.legend(loc='best')


