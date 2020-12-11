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
import os
from nogoData import nogo_turn
from collections import defaultdict


def import_data():
    f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\active_mice')
    d = h5py.File(f,'r')
    return d

def import_data_multi(mice, fold):
    '''mice is a dict of mice and dates you want to average
    fold is the folder it's in, ex='exp_opto' or 'training_to_analyze'
    ex = {'525455':'1122', '525458':'1024'}
    returns list of hd5 files ready for indexing, analyzing, etc'''
    
    
    directory = r'\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\active_mice'
    files = []
    
    for m in mice:
        dataDir = os.path.join(os.path.join(directory, m), fold + '/')   
        all_files = os.listdir(dataDir)
        for f in all_files:
            if mice[m] in f:
                files.append(dataDir+f)
                
    hd5files = [h5py.File(i, 'r') for i in files]
    
    return hd5files



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


def ignore_after(data, lim):
    d = data
    if type(lim) == int:      # how many consecutive trials to stop after
        count = 0
        for i, resp in enumerate(d['trialResponse'][:]):
            if resp==0:
                count+=1
            elif resp==-1 or resp==1:
                count=0
            else:
                count+=0  # ignore catch
            
            if count == lim:
                trialNum=i-lim
                break
            else:
                trialNum=i
                
        startFrame = d['trialStartFrame'][trialNum]
        
    elif lim == list:       # to ignore after a certain time or a selection of session - use minutes
        start, end  = lim[0], lim[1]
        # convert minutes into frames;  match frames to trial num;  index by trial num
        # need to finish this
    
    return trialNum, startFrame-1



def dates_to_dict(task=None):  #use with combined plotting


    df = pd.read_excel(r'\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Masking training notes new.xlsx', 
                   sheet_name='Task_vals')

    df.index = df['mouse_id']

    d = df.to_dict()   # turn df into dict by task type, {mouse:date}
    
    selection = d[task]
    
    new_d = {str(key): str(value) for key, value in selection.items()}
    
    return new_d



def create_df(data):   
    
## PULL RELEVANT DATA FROM HDF5 FILE TO CREATE DATAFRAME 
    
    d = data
    mouse, date = str(d).split('_')[-3:-1]
        
    trialResponse = d['trialResponse'][:-1]
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
    
    #quiescent period violations
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
    deltaWheel = [deltaWheelPos[start:stim+(openLoop*2)+maxResp+postReward] for (start, stim, openLoop) in  
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
    
    if 'trialType' in d.keys():
        df['trialType'] = d['trialType'][:end] 

    df['trialLength_ms'] = convert_to_ms(trialLength)
    df['closedLoopFramesTotal'] = trialResponseFrame-trialStimStartFrame-trialOpenLoopFrames
    
    df['targetDuration'] = convert_to_ms(trialTargetFrames)
    df['targetContrast'] = trialTargetContrast
    
    
    def fill():
        return np.zeros(len(trialResponse)).astype(int)

# if masking    
    if d['probMask'][()] > 0:
        #df['maskLength_ms'] = convert_to_ms(d['trialMaskFrames'][:end])
        df['maskContrast'] = trialMaskContrast
        df['maskOnlyMove'] = fill()

        for col in 'maskOnlyMove':
            for (i,turn) in zip(inds[1], turns[1]):
                df.at[i, col] = turn
                
        df['soa'] = trialMaskOnset
        df['soa_frames'] = d['trialMaskOnset'][:len(df)]  

# if nogo trials
#    if d['probNoGo'][()]>0:
#        df['nogo'] = False
#        for i in nogos:
#            df.loc[i, 'nogo'] = True
#            df.loc[i, 'soa'] = -maskOnset[0]  # this helps for summary stats
#        df['nogoMove'] = fill()
#
#        for col in 'noGoMove':
#                for (i,turn) in zip(inds[0], turns[0]):
#                    df.at[i, col] = turn
        
# if using optogenetics       
    if 'probOpto' in d and d['probOpto'][()]>0:
        df['optoOnset'] = d['trialOptoOnset'][:len(df)]

        
    df['repeat'] = repeats    
    
    df['quiescentViolations'] = fill()
    for key,val in qDict.items():
        df.at[key, 'quiescentViolations'] = len(val)
        
    df['deltaWheel'] = deltaWheel
    
    df['trialFrameIntervals'] = frames
 
#dataframe metadata
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
    
    df['catchTrial'] = ~np.isfinite(df['rewDir'])

    
    return df



def get_dates(dataframe):
    
    from datetime import datetime
    import pandas
    df=dataframe
    if type(df) is pandas.core.frame.DataFrame:
        if type(df.date) is list:
            dates = [datetime.strptime(date, '%Y%m%d').strftime('%m/%d/%Y') for date in df.date]
            date = '-'.join([dates[0], dates[-1]])
        else:
            date = datetime.strptime(df.date, '%Y%m%d').strftime('%m/%d/%Y')
    elif type(df) is str:   #i.e. not a dataframe, a file
        date = datetime.strptime(df, '%Y%m%d').strftime('%m/%d/%Y')
    else:
        print('Wrong Format for date')
        
    return date




def rxnTimes(data, dataframe, version=None):
    
    d = data
    df = dataframe
            
    monitorSize = d['monSizePix'][0] 
    maxResp = d['maxResponseWaitFrames'][()]
    postReward = d['postRewardTargetFrames'][()]
    
# these 2nd values are needed for files after 6/18/2020   
    # reward dist and queiscentMove in mm
    rewardDist = d['wheelRewardDistance'][()] if 'wheelRewardDistance' in d.keys()\
        else d['normRewardDistance'][()]  #normalized to screen width in pixels- used for wheelgain
    maxQuiescentMove = d['maxQuiescentNormMoveDist'][()] if 'maxQuiescentNormMoveDist' in d.keys()\
        else d['maxQuiescentMoveDist'][()]   # in mm
    wheelGain = d['wheelSpeedGain'][()] if 'wheelSpeedGain' in d.keys() else d['wheelGain'][()]
    
    wheelRadius = d['wheelRadius'][()]
    
    if 'wheelRewardDistance' in d.keys():
        #initiationThresh = rewardDist/wheelRadius
        initiationThresh = .2
    else:
        initiationThreshDeg = 0.5 
        initiationThresh= initiationThreshDeg*np.pi/180*maxQuiescentMove  
        
    sigThreshold = maxQuiescentMove if 'wheelRewardDistance' in d.keys() else maxQuiescentMove*monitorSize  
    rewThreshold = rewardDist if 'wheelRewardDistance' in d.keys() else rewardDist*monitorSize

    fi = d['frameIntervals'][:]
    if len(np.unique(df['openLoopFrames'])) == 1:
        closedLoop = int(np.unique(df['openLoopFrames'])*(1000/df.framerate))
    else:
        closedLoop = df['openLoopFrames']/df.framerate
    
    stimInds = df['stimStart'] - df['trialStart']

    interpWheel = []
    initiateMovement = []
    significantMovement = []
    ignoreTrials = []  # old ignore_trials func actually calls current func and returns this list
    outcomeTimes = []
    
    check =[]
    
    ## use below code to determine wheel direction changes during trial 
    # during just trial time (ie in nogos before trial ends) or over entire potential time? both?

 
    for i, (resp, rew, fiInd, stimInd, openLoop) in enumerate(zip(
            df['resp'], df['rewDir'], df['stimStart'], stimInds, df['openLoopFrames'])):

        wheel = df.loc[i, 'deltaWheel'].copy()
        
        f = fi[fiInd:(fiInd+(len(wheel)-stimInd-postReward))]   #from stim start frame to len of maxTrial
        wheel *= wheelRadius  #(180/np.pi * wheelRadius)   # in frames 
        fp = np.cumsum(wheel[stimInd:stimInd+len(f)])   

        xp = np.cumsum(f)    
        x = np.arange(0, xp[-1], .001)   # in ms 
        interp = np.interp(x,xp,fp)
        interpWheel.append(interp)
      
        sigMove = np.argmax(abs(interp)>=sigThreshold)   
        significantMovement.append(sigMove)
        if 0<sigMove<100:
            ignoreTrials.append(i)
       
## outcome times
#        if rew>0:
        outcome = np.argmax(abs(interp[closedLoop:]) >= (rewThreshold + abs(interp[closedLoop]))) + closedLoop
        
        if outcome==closedLoop:
            outcome = 0
        outcomeTimes.append(outcome)

        
        if version=='sam':
            if outcome > 0:
                trace = interp[:outcome:-1]
                
            
        elif version=='corbett':
            q = d['quiescentFrames'][()]
            qperiod = np.cumsum(wheel[stimInd-q:stimInd])
            mvmt = np.percentile(abs(qperiod), 99)
            init = np.argmax(abs(interp)>(mvmt))
            initiateMovement.append(init)
            print(i, mvmt, init)
            
        else:
     
         ## intitiation times   
            if (rew==0) and (resp==1):   #catch/nogo trials 
                init = 0 
                
            else:
                init = np.argmax(abs(interp)>initiationThresh)
                 
            
    #        if (0<init<150) and sigMove>150:
    #            init = np.argmax(abs(interp[150:])>(initiationThresh + interp[150])) + 150
                # maybe change this to just send to ignore - right now ignoring early move
            if sigMove-init>100:
                check.append(i)
                init = np.argmax(np.round(abs(interp[0:sigMove]),3)>.25)
            
            if (init==0) and (resp==1):
                init = np.argmax(np.round(abs(interp[0:outcome]),3)>.25)
               
            elif init<100 and sigMove>1:
                init = np.argmax(np.round(abs(interp[0:sigMove]),3)>.25)
                if init<100:
                    ignoreTrials.append(i)
                    
            initiateMovement.append(init)
        
        # also want time from start of choice til choice  (using modified version of sam's method)
                 
       

    return np.array([initiateMovement, outcomeTimes, ignoreTrials])


# code to plot the above wheel traces, to visually inspect for accuracy
#test = [i for i, e in enumerate(interpWheel[:100]) if type(e)!=int]
#    
## --- for catch trials ---
#catchTrials = [i for i, row in df.iterrows() if row.isnull().any()]
#
for i, x in enumerate(initiateMovement[:20]):
    if (x>0) and (i not in ignoreTrials): #and (df.iloc[i]['resp']==1):
        plt.figure()
        plt.plot(interpWheel[i])
        plt.title('Reward ' + df.loc[i, 'rewDir'].astype(str) + '  , Response ' + df.loc[i, 'resp'].astype(str) + '     ' + str(i))
        plt.vlines(closedLoop, -10, 10, color='g', ls ='--', label='Closed Loop', lw=2)
        plt.vlines(initiateMovement[i], -10, 10, ls='--', color='m', alpha=.4, label='Initiation')
        plt.vlines(significantMovement[i], -10, 10, ls='--', color='c', alpha=.4 , label='Q threshold')
        plt.vlines(outcomeTimes[i], -10, 10, ls='--', color='b', alpha=.3, label='Outcome Time')
        plt.vlines(df['trialLength_ms'][i], -10, 10, label='Trial Length')
        plt.ylim([-10,10])
        plt.legend(loc='best', fontsize=10)
#        
#
### ---- for ignoreTrials ----
#    
#for i in ignoreTrials:
#    fig, ax = plt.subplots()
#    plt.plot(interpWheel[i], color='k', alpha=.5)
#    plt.suptitle(i)
#    plt.title('Reward ' + df.loc[i, 'rewDir'].astype(str) + '  , Response ' + df.loc[i, 'resp'].astype(str))
#    plt.vlines(150, -40, 40, color='g', ls ='--', label='Closed Loop', lw=2)
#    plt.vlines(initiateMovement[i], -40, 40, ls='--', color='m', alpha=.4, label='Initiation')
#    plt.vlines(significantMovement[i], -40, 40, ls='--', color='c', alpha=.4 , label='Q threshold')
#    plt.vlines(outcomeTimes[i], -40, 40, ls='--', color='b', alpha=.3, label='Outcome Time')
#    plt.vlines(df['trialLength_ms'][i], -50, 50, label='Trial Length')
#    ax.set_xlim([0, closedLoop])
#    ax.set_ylim([-6,6])
#    plt.legend(loc='best')

