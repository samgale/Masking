# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:22:58 2020

@author: svc_ccg

takes behavioral data from specified mice and plots an averaged response rate
and percent correct; returns 2 plots
Declare task and mice in function call
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plotting_variable_params import plot_param
from opto_masking import plot_opto_masking
from types import SimpleNamespace    
from dataAnalysis import import_data_multi


def plot_average_beh(task=None, mice, info):
    '''
    info is a dict of mouse_ids and dates - ex: {'525458':'1022', '516686':'1015'}
    task is the task type you want to average (e.g., 'opto masking')
    '''
    
    # for each mouse listed, pull file and get data
    # OR - run task-specific plotting code but get arrays back (this reduces code and helps w indexing)
    
    if 'opto' in task:
         
        folder = 'exp_opto'
        
        if task == 'opto masking':
            func = plot_opto_masking
            
        elif task == 'opto unilateral' or task='opto contrast':
            func = plot_param          # this returns counts not percents 
            
        task == None
            
    else:
        
        func = plot_param
        
        if task == 'opto unilateral':   #will have 4 lines 
            folder = 'exp_opto'
        
        if task == 'target duration':  # 2 plots of a single percent
            task = 'targetLength'
            folder = 'exp_duration'
    
        elif task == 'target contrast':
            folder = 'exp_contrast'
        
        elif task == 'masking':
            folder = 'exp_masking'

        else:
            print('No task specified')


    f = import_data_multi(mice, folder)  # list of the hdf5 files for each mouse 
    
    percents = []   #each item in percents has mouseID, dict of param vals, then dicts of vals to plot
    
    for i in f:
        percents.append(func(i, param=task, array_only=True))  # calls function appropriate to params and returns array of %s
        
            ## need to think about mask only 
        
#    for e, x in enumerate(percents[1:]):
#        for key, val in percents[e].items:
#           print(key, val)
    # in each dict only the 1st 2 keys are averages 
    
    for i in range(len(mice)):
        for key, val in percents[i][1].items():
            if key == 'Fraction Correct' or key=='Response Rate':
                print(key, val)
     
        print(np.nanmean([percents[1][1][key], percents[0][1][key]], axis=0))  #averaged over mice for eac hkey (resp and corr)
        


        
        
    for i in range(len(mice)):
        for key, val in percents[i][1].items():
            
            if key == 'Fraction Correct' or key=='Response Rate':
            
                fig, ax = plt.subplots()
                ax.plot(val)
                plt.title(key)
            
        
        
        
        
        
        
 