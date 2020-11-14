# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:24:49 2020

@author: svc_ccg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plotting_variable_params import plot_param
from opto_masking import plot_opto_masking
from types import SimpleNamespace    
from dataAnalysis import import_data_multi, dates_to_dict, create_df
from behaviorAnalysis import formatFigure
from plotting_opto_unilateral import plot_opto_uni
import scipy.stats
import seaborn as sns

def plot_separate_sides(task=None):
    
########## THIS ONLY WORKS WITH MASKING
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    sns.set_style('white')

    # for each mouse listed, pull file and get data
    # run task-specific plotting code but get arrays back (this reduces code and helps w indexing)
    
    if 'opto' in task:
         
        folder = 'exp_opto'
        
        if task == 'opto masking':
            param = task
            func = plot_opto_masking
            
        elif task == 'opto unilateral':
            func = plot_opto_uni
            param = None
        
        elif task=='opto contrast':
            func = plot_param          
            param='opto'
            
        lbl = 'Optogenetic Light Onset Relative to Target Onset (ms)'    
        task == None
            
    else:
        
        func = plot_param
        
        if task == 'masking':  # ^^
            param = 'soa'
            folder = 'exp_masking'
    
        else:
            print('No task specified')
            
            
    
    mice = dates_to_dict(task=task)
    files = import_data_multi(mice, folder)  # list of the hdf5 files for each mouse 
    
    
    
    percents = []   #each item in percents has mouseID, dict of param vals, then dicts of vals to plot
    
    for f in files:    # calls function appropriate to params and returns array of %s  
        percents.append(func(f, param=param, ignoreNoRespAfter=15, returnCounts=True)) 
        
    rn = range(len(mice))        


## RESPONSE RATE
      
    rightMaskRespRate = []
    leftMaskRespRate = []

    for i in rn:
        x = list(percents[i][1].values()) 
        rightMaskRespRate.append(x[4][0][0]/x[1][0][0])   #17 ms SOA L (resps/totals)
        leftMaskRespRate.append(x[4][1][0]/x[1][1][0])  # 17 ms SOA
    
    rightTargetOnlyRespRate = []
    leftTargetOnlyRespRate = []

    for i in rn:
        x = list(percents[i][1].values()) 
        rightTargetOnlyRespRate.append(x[4][0][4]/x[1][0][4])   # target only R (resps/totals)
        leftTargetOnlyRespRate.append(x[4][1][4]/x[1][1][4])  # target only L turn

    fig, ax = plt.subplots(figsize=(7,7))
    plt.scatter(rightMaskRespRate, leftMaskRespRate, color='k', s=24, label='17 ms SOA mask')
    plt.scatter(rightTargetOnlyRespRate, leftTargetOnlyRespRate, color='c',  s=24, label='Target only')
    ax.plot([0,1.05], [0, 1.05], '--', color='k', alpha=.3)
    plt.title('Response Rate Right vs Left Turning Trials')
    plt.xlabel('Response Rate, Right Turning')
    plt.ylabel('Response Rate, Left Turning')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0,1.05])
    ax.set_aspect('equal')
    plt.legend(fontsize=10, loc='best', numpoints=1)
    formatFigure(fig, ax)



## FRACTION CORRECT   
    leftMaskFracCorr = []
    rightMaskFracCorr = []
    
    for i in rn:
        x = list(percents[i][1].values()) 
        leftMaskFracCorr.append(x[2][0][0]/x[4][0][0])   #17 ms SOA   (hits/resps)
        rightMaskFracCorr.append(x[2][1][0]/x[4][1][0])  # 17 ms SOA
    
    leftTargetOnlyFracCorr = []
    rightTargetOnlyFracCorr = []
    
    for i in rn:
        x = list(percents[i][1].values()) 
        leftTargetOnlyFracCorr.append(x[2][0][4]/x[4][0][4])   # target only  (hits/resps)
        rightTargetOnlyFracCorr.append(x[2][1][4]/x[4][1][4])  # target Only
    
    fig, ax = plt.subplots(figsize=(7,7))
    plt.scatter(rightMaskFracCorr, leftMaskFracCorr, color='k',  s=24, label='17 ms SOA mask')
    plt.scatter(rightTargetOnlyFracCorr, leftTargetOnlyFracCorr, color='c', s=24, label='Target only')
    ax.plot([0,1.05], [0, 1.05], '--', color='k', alpha=.3)
    plt.title('Fraction Correct Right vs Left Turning Trials')
    plt.xlabel('Fraction Correct, Right Turning')
    plt.ylabel('Fraction Correct, Left Turning')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0,1.05])
    plt.legend(fontsize=10, loc='best', numpoints=1)
    formatFigure(fig, ax)