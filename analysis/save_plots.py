# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:59:57 2020

@author: chelsea.strawder
"""

import behaviorAnalysis
import performanceBySOA
from plotting_variable_params import plot_param
from SessionPerformance import plot_session
from responsePlotByParam import plot_by_param
import qualityControl
from catchTrials import catch_trials
import matplotlib.pyplot as plt
import os
from plottingOptoAgainstParam import plot_opto_vs_param
from opto_masking import plot_opto_masking
from plotting_opto_unilateral import plot_opto_uni


def save_daily_plots(data, INR):
    
    d = data
    plt.ioff()
    
    mouse_id=d['subjectName'][()]
    date = d['startTime'][()].split('_')[0][-4:]
    date = date[:2]+'-'+date[2:]
    
    date = date if date[:2] in ['10','11','12'] else date[-4:]
    
    ignoreNoResp = INR  # change this if necessary
    
    directory = r'\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\active_mice'

    dataDir = os.path.join(os.path.join(directory, mouse_id), 'Plots') 
    wheelDir = os.path.join(dataDir, 'Wheel plots')
    
    
# daily wheel plot
    behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], ignoreRepeats=True, 
                                   ion=False, framesToShowBeforeStart=0, mask=False, 
                                   maskOnly=False, xlim='auto', ylim='auto', ignoreNoRespAfter=ignoreNoResp)
    
    plt.savefig(wheelDir+'/Daily Wheel/' + mouse_id + ' ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plot no response trials only (with repeats)
    behaviorAnalysis.makeWheelPlot(d, responseFilter=[0], ignoreRepeats=False, ion=False, 
                                   framesToShowBeforeStart=0, mask=False, maskOnly=False,  
                                   xlim='auto', ylim=[-8,8], ignoreNoRespAfter=ignoreNoResp )
        
    plt.savefig(wheelDir+'/No Resp Wheel/' + mouse_id + ' ' + date + ' no resp.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plots catch trial wheel traces 
   
    catch_trials(d, xlim='auto', ylim='auto', plot_ignore=False, arrayOnly=False, ion=False) 
    
    plt.savefig(wheelDir+'/Catch/' + mouse_id + ' catch ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plot activity over entire session, trial-by-trial - 1 plot
    plot_session(d, ion=False, ignoreNoRespAfter=ignoreNoResp)
    plt.savefig(dataDir + '/Session plots/' + mouse_id + ' session ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plots frame distribution and frame intervals 
    qualityControl.check_frame_intervals(d)
    
    plt.savefig(dataDir + '/Other plots/frame intervals/' +  
                'frame intervals ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.savefig(dataDir + '/Other plots/frame dist/' +  
                'frame dist ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# check number of quiescent period violations - use 'sum' for cumsum OR 'count' for count per trial
    qualityControl.check_qviolations(d, plot_type='sum')
    
    plt.savefig(dataDir + '/Other plots/quiescent violations/' +  
                'Qvio ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    qualityControl.check_qviolations(d, plot_type='count')
    
    plt.savefig(dataDir + '/Other plots/quiescent violations/' +  
                'Qvio ' + date + ' count.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
# check distribution of delta wheel position 
    qualityControl.check_wheel(d)
    plt.savefig(dataDir + '/Other plots/wheel pos/' +  
                'wheel ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()


## additional plots 
    if d['moveStim'][()]==False:
        
        def add_folder(loc):
            newFolder = date.replace('/','-')
            path = os.path.join(os.path.join(dataDir, loc), newFolder)
            if not os.path.exists(path):
                os.mkdir(path)
            return path
        
        def save_new_plots(files, new_path):
            for f in files:
        
                plt.savefig(new_path + '/' + mouse_id + f + date + '.png', dpi=300)
                plt.close()
            
        
        if len(d['targetFrames'][:])>1:
            
            param = 'targetLength'   # target duration exps
            
            if d['probOpto'][()]==0:   #defining param first allows us to select which opto plotting to use later
                
                plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=ignoreNoResp)  # creates 3 plots
                
                file_names = [' target duration response rate ', ' target duration correct given response ',
                              ' target duration fraction correct ']
                
                save_new_plots(file_names, add_folder('Duration plots'))
                

        
        if len(d['targetContrast'][:])>1:
            
            param = 'targetContrast'   # target contrast exps
            
            if d['probOpto'][()]==0:

                plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=ignoreNoResp)  # creates 3 plots
                
                file_names = [' target contrast response rate ', ' target contrast correct given response ',
                              ' target contrast fraction correct ']
                
                save_new_plots(file_names, add_folder('Contrast plots'))
                

            
        
        if len(d['maskOnset'][:])>1 or d['probMask'][()] > 0:
          
            param = 'soa'   # masking exps
        
            
            if d['probOpto'][()]==0:
                
                plot_param(d, param='soa', ignoreNoRespAfter=ignoreNoResp)   # creates 3 plots
                
                file_names = [' masking response rate ', ' masking correct given response ',
                              ' masking fraction correct ']
                
                save_new_plots(file_names, add_folder('Masking plots'))


  
    if d['probOpto'][()] > 0:
        if len(d['optoChan'][:]) > 1:
            
             plot_opto_uni(d, ignoreNoResp)
             file_names = [' unilateral opto ', ' catch trials ']
        else:
        
            try: param
            except NameError: param = 'opto'  # if param not defined by above conditional, must be opto
            
            if param=='targetContrast' or param=='targetLength':    # opto contrast exps
                
                plot_opto_vs_param(d, param, ignoreNoRespAfter=ignoreNoResp, plotType='single')
                file_names = [' combined param correct ', ' combined opto correct ', 
                              ' combined param response ', ' combined opto response ']
                
            elif param=='soa':  # opto masking exps
                
                plot_opto_masking(d)    
                file_names = [' opto masking fraction correct ', ' opto masking response rate ']
                
            else:   # opto 
                
                plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=None)
                file_names = [' opto response rate  ', ' opto correct ']


        save_new_plots(file_names, add_folder('Opto plots'))
            
        

        
