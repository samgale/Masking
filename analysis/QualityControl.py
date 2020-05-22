# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:31:42 2020

@author: chelsea.strawder

sanity checking and quality control after trials 

"""

from dataAnalysis import import_files, create_df
import numpy as np
import matplotlib.pyplot as plt



# checks the frame intervals from stim onset to mask onset
def check_soa_frames(df):
    '''
    Compares the specified mask onset time with the actual frame intervals during
    that portion of the trial.  Marks certain trials for exclusion if the intervals 
    exceed a set threshold 
    '''
    #select the trials that have visible masks
    actualMask = np.array([t for t in df['actualSOA_ms'] if t>0])
    maskTheory = np.array([m*(1000/df.framerate) for m in df['soa_frames'] if m > 0])
        
    diffs = np.array(maskTheory - actualMask)
    diffs_opp = np.array(actualMask - maskTheory)/1000
    
    # theory - actual
    df['onsetDiff_ms'] = (df['actualSOA_ms'] - df['soa_frames']*1000/df.framerate)
    
    fig, ax = plt.subplots()
    
   # plt.scatter(maskTheory, actualMask)
    plt.hist(diffs_opp, edgecolor='k', linewidth=1, bins = np.arange(0, max(fi)+1/fr, 1/fr) - 0.5/fr)
    plt.xlabel('Specified SOA, in ms')
    plt.ylabel('Actual SOA, in ms')
    #ax.set_yscale('log')
    #ax.set_xticks(np.round(np.unique(maskTheory)))
    plt.title('Comparing specified vs actual mask onset')
    plt.suptitle(df.mouse + '  ' + df.date)

    
# check for dropped frames    
def check_frame_intervals(d):
    '''
    plots distribution of frame intervals over the session
    with hist bins centered at common frame intervals (.008, .016, etc)
    assuming a framerate of 120 hz
    '''
    
    fi = d['frameIntervals'][:]   # in seconds 
    fr = int(np.round(1/np.median(fi)))   # frames per second
    
    fig, ax = plt.subplots()
    
    plt.hist(fi, edgecolor='k', linewidth=1, bins = np.arange(0, max(fi)+1/fr, 1/fr) - 0.5/fr)
    ax.set_yscale('log')
    ax.set_xticks(np.round(np.arange(0, max(fi), 1/fr), 3))
    ax.tick_params(axis='x', rotation=60)
    plt.title('Distribution of Frame Intervals')
    ax.set_ylabel('Trials')
    ax.set_xlabel('Frame Intervals (sec)')
    mouse, date = str(d).split('_')[-3:-1]
    plt.suptitle(mouse + '  ' + date)



# visual overview of when high frame intervals are occuring in the session
def dropped_frames(df):
    max_fi = [max(t) for t in df['trialFrameIntervals']]
  
    plt.figure()
    plt.plot(max_fi)
    for i, f in enumerate(max_fi):
        if f>.1:
            print(i, f)





