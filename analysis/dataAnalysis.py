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
    df.mouse = {val.mouse for val in dict1.values()}
    df.date = [val.date for val in dict1.values()]
    return df


def create_df(d):   
    
##pull all of the relevant data to create a dataframe object 
    
    mouse, date = str(d).split('_')[-3:-1]
    
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))
    
    def convert_to_ms(value):
        return value * 1000/framerate
    
    trialResponse = d['trialResponse'][:]
    end = len(trialResponse)
    trialRewardDirection = d['trialRewardDir'][:end]
    trialTargetFrames = d['trialTargetFrames'][:end]
    trialTargetContrast = d['trialTargetContrast'][:end]
    
    trialOpenLoopFrames = d['trialOpenLoopFrames'][:end]
    if len(np.unique(trialOpenLoopFrames)>1):
        pass
        
    #    preStimFrames = d['preStimFramesFixed'][()]
    #    preStimVar = d['preStimFramesVariableMean'][()]              
    #    openLoopFrames = d['openLoopFramesFixed'][()]
    #    openLoopVar = d['openLoopFramesVariableMean'][()]
    #    openLoopMax = d['openLoopFramesMax'][()]
      
    trialStartFrame = d['trialStartFrame'][:end]
    trialStimStartFrame = d['trialStimStartFrame'][:]
    trialResponseFrame = d['trialResponseFrame'][:end] 
    quiescentMoveFrames = [q for q in d['quiescentMoveFrames'][:] if q<trialStimStartFrame[-1]]
    
    maxResp = d['maxResponseWaitFrames'][()]
    deltaWheelPos = d['deltaWheelPos'][:]                      
    repeats = d['trialRepeat'][:end]
    #nogoWait = d['nogoWaitFrames'][()]
        
    maskOnset = convert_to_ms(d['maskOnset'][()])
    trialMaskOnset = convert_to_ms(d['trialMaskOnset'][:end])
    trialMaskContrast = d['trialMaskContrast'][:end]

    
### process & clean data
    
    for i, target in enumerate(trialTargetFrames):  # this is needed for older files nogos are randomly assigned a dir
        if target==0:
            trialRewardDirection[i] = 0
    
    nogos = [i for i, (rew, con) in enumerate(zip
             (trialRewardDirection, trialMaskContrast)) if rew==0 and con==0]
   
    if np.any(trialMaskOnset>0):
        targetOnlyVal = maskOnset[-1] + 0.5*maskOnset[-1] #round(np.mean(np.diff(maskOnset)))  # assigns targetOnly condition an evenly-spaced value from soas
        maskOnset = np.append(maskOnset, targetOnlyVal)                     # makes final value the targetOnly condition
            
        for i, (mask, trial) in enumerate(zip(trialMaskOnset, trialTargetFrames)):   # filters target-Only trials 
            if trial>0 and mask==0:
                trialMaskOnset[i]=targetOnlyVal      
    
    trialLength = trialResponseFrame - trialStimStartFrame

    #gives entire wheel trace from trial start to the max Length of trial
    deltaWheel = [deltaWheelPos[start:stim+openLoop+maxResp] for (start,stim, openLoop) in 
                  zip(d['trialStartFrame'][:len(trialStimStartFrame)], trialStimStartFrame, trialOpenLoopFrames)]
    
    turns, inds = nogo_turn(d)      #for both of these, [0]=nogos, [1]=maskOnly                    
    
    
    qDict = defaultdict(list)
    for i, (start,stimStart) in enumerate(zip(trialStartFrame, trialStimStartFrame)):
        for x in quiescentMoveFrames:    
            if start<x<stimStart:
                qDict[i].append(x)
          
    assert len(quiescentMoveFrames) == sum([len(qDict[x]) for x in qDict]), "Qframes Error"
                      
### Create dataframe
                
    data = list(zip(trialRewardDirection, trialResponse, 
                    trialStartFrame, trialStimStartFrame, trialResponseFrame, trialOpenLoopFrames))

    index = np.arange(len(trialResponse))
    df = pd.DataFrame(data, 
                      index=index, 
                      columns=['rewDir', 'resp', 'trialStart', 'stimStart', 'respFrame', 'openLoopFrames'])
    
    df['trialLength'] = convert_to_ms(trialLength)
    
    df['mask'] = trialMaskContrast
    df['soa'] = trialMaskOnset
    #df['maskLength'] = convert_to_ms(d['trialMaskFrames'][:end])
    df['maskContrast'] = trialMaskContrast

    df['targetDuration'] = convert_to_ms(trialTargetFrames)
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
    
    
    df.mouse = mouse
    df.date = date
    
    times = rxnTimes(d, df)  # 0==initiation, 1==outcome, 2==ignore
    
    df['initiationTime'] = times[0]
    df['outcomeTime'] = times[1]
    df['ignoreTrial'] = False   
    for i in times[2]:
        df.loc[i, 'ignoreTrial'] = True
    
    return df


def wheel_trace_slice(dataframe):
    df = dataframe

    wheelDF = df[['trialStart','stimStart', 'respFrame', 'deltaWheel']].copy()
    wheelDF['wheelLen'] = list(map(len, wheelDF.loc[:,'deltaWheel']))
    
    wheelDF['diff1'] = wheelDF['stimStart'] - wheelDF['trialStart']  #prestim
    wheelDF['diff2'] = wheelDF['respFrame'] - wheelDF['trialStart']  #entire trial
    
    wheel = [wheel[start:stop] for (wheel, start, stop) in zip(
            wheelDF['deltaWheel'], wheelDF['diff1'], wheelDF['wheelLen'])]
    
    return wheel


def rxnTimes(data, dataframe):
    
    d = data
    df = dataframe
    
    fi = d['frameIntervals'][:]
    framerate = int(np.round(1/np.median(fi)))
    
    monitorSize = d['monSizePix'][0] 
    
    normRewardDist = d['normRewardDistance'][()]
    maxQuiescentMove = d['maxQuiescentNormMoveDist'][()]
    wheelSpeedGain = d['wheelSpeedGain'][()]
    
    initiationThreshDeg = 0.5  #how did he decide this?
    initiationThreshPix = initiationThreshDeg*np.pi/180*wheelSpeedGain
    sigThreshold = maxQuiescentMove * monitorSize
    rewThreshold = normRewardDist * monitorSize

    wheelTrace = wheel_trace_slice(df)
    cumulativeWheel = [np.cumsum(mvmt) for mvmt in wheelTrace]

    interpWheel = []
    initiateMovement = []
    significantMovement = []
    ignoreTrials = []  # old ignore_trials func actually calls this func, returns this list
    outcomeTimes = []
    
    ## use below code to determine wheel direction changes during trial 
    # during just trial time (ie in nogos before trial ends) or over entire potential time? 
    
    for i, (wheel, resp, rew, soa) in enumerate(zip(
            cumulativeWheel, df['resp'], df['rewDir'], df['soa'])):

        fp = wheel
        xp = np.arange(0, len(fp))*1/framerate
        x = np.arange(0, xp[-1], .001)
        interp = np.interp(x,xp,fp)
        interpWheel.append(interp)
        
        sigMove = np.argmax(abs(interp)>=sigThreshold)
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
        outcome = np.argmax(abs(interp)>= rewThreshold)
        if outcome>0:
            outcomeTimes.append(outcome)
        else:
            outcomeTimes.append(0)

    return np.array([initiateMovement, outcomeTimes, ignoreTrials])

## code to plot the above wheel traces, to check for accuracy
#test = [i for i, e in enumerate(interpWheel) if type(e)!=int]
#
#for i in test[:40]:
#    plt.figure()
#    plt.plot(interpWheel[i], lw=2)
#    plt.title(i)
#    plt.vlines(initiateMovement[i], -400, 400, ls='--', color='m', alpha=.4)
#    plt.vlines(significantMovement[i], -400, 400, ls='--', color='c', alpha=.4 )
#    plt.vlines(outcomeTimes[i], -400, 400, ls='--', color='b', alpha=.3)


