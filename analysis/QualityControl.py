# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:31:42 2020

@author: chelsea.strawder

sanity checking and quality control after trials 

one thing to consider is that the longer SOAs have a greater chance of having 
dropped frames

from what I can tell, major dropped frames are happening at the start of the mask,
or right before the mask comes on

"""

from dataAnalysis import import_data, create_df
import numpy as np
import matplotlib.pyplot as plt

d = import_data()
df = create_df(d)

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
        
    diffs = np.array(actualMask - maskTheory)/1000   # in seconds
    
    # theoretical - actual
    df['onsetDiff_ms'] = (df['actualSOA_ms'] - df['soa_frames']*1000/df.framerate)
    
    ### exclude trials where the actual SOA is longer than the specified 
    # SOA by >0.5/120 seconds (HALF A FRAME)
    for i, x in enumerate(df['onsetDiff_ms']):
        if abs(x) > (.5/df.framerate)*1000:
            print(i)
            df.loc[i, 'ignoreTrials'] = True
    
      
    fig, ax = plt.subplots()
    fr = df.framerate
    plt.hist(diffs, edgecolor='k', linewidth=1, bins = np.arange(0, max(diffs)+1/fr, .25/fr) - 0.5/fr, 
             density=True, stacked=True)
    plt.xlabel('Difference in seconds')
    plt.ylabel('Count')
    ax.set_yscale('log')
    plt.title('Comparing actual vs specified mask onset')
    plt.suptitle(df.mouse + '  ' + df.date)

    
# check for dropped frames    
def check_frame_intervals(d):
    '''
    plots distribution of frame intervals over the session
    with hist bins centered at common frame intervals (.008, .016, etc)
    assuming a framerate of 120 hz
    '''
    
    ### FREQUENCY RATHER THAN RAW COUNT - SAM
    
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
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])


# visual overview of when high frame intervals are occuring in the session
def dropped_frames(df):
    
    max_fi = [max(t) for t in df['trialFrameIntervals']]
  
    plt.figure()
    plt.plot(max_fi)
    plt.ylabel('Max Frame Interval, in sec')
    plt.xlabel('Trial Number')
    plt.title('Max Frame Intervals Per Trial')
    plt.suptitle(df.mouse + '  ' + df.date)
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])


    fi_inds = [i for i, f in enumerate(max_fi) if f>.05]
    print((np.diff(fi_inds)))   # difference btwn trials with high frame intervals
    above_t = [max_fi[y] for y in fi_inds]  
    





