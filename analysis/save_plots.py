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



def save_daily_plots(data):
    
    d = data
    plt.ioff()
    
    mouse_id=d['subjectName'][()]
    date = d['startTime'][()].split('_')[0][-4:]
    date = date[:2]+'-'+date[2:]
    
    date = date if date[:2] in ['10','11','12'] else date[-4:]
    
    ignoreNoResp = 10   # change this if necessary
    
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
        
        if len(d['targetFrames'][:])>1:
            param = 'targetLength'
            
            if d['probOpto'][()]==0:
            
                plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=ignoreNoResp)  # creates 3 plots
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target duration response rate ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target duration correct given response ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target duration fraction correct ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        
        if len(d['targetContrast'][:])>1:
            param = 'targetContrast'
            
            if d['probOpto'][()]==0:

                plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=ignoreNoResp)  # creates 3 plots
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target contrast response rate ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target contrast correct given response ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Other plots/other/' + mouse_id + 
                            ' target contrast fraction correct ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        
        if len(d['maskOnset'][:])>1 or d['probMask'][()] > 0:
            param = 'soa'
            
            if d['probOpto'][()]==0:

                plot_param(d, param='soa', ignoreNoRespAfter=ignoreNoResp)   # creates 3 plots
        
                plt.savefig(dataDir + '/Masking plots/' + mouse_id + 
                            ' masking response rate ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Masking plots/' + mouse_id + 
                            ' masking correct given response ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.savefig(dataDir + '/Masking plots/' + mouse_id + 
                            ' masking fraction correct ' + date + '.png', dpi=300, bbox_inches='tight')
                plt.close()

    if d['probOpto'][()] > 0:
        
        newFolder = date.replace('/','-')
        path = os.path.join(os.path.join(dataDir, 'Opto plots'), newFolder)
        if not os.path.exists(path):
            os.mkdir(path)
        
        if param=='targetContrast' or param=='targetLength':
            plot_opto_vs_param(d, param, ignoreNoRespAfter=ignoreNoResp, plotType='single')
            file_names = [' combined param correct ', ' combined opto correct ', 
                          ' combined param response ', ' combined opto response ']
            
        elif param=='soa':
            plot_opto_masking(d)
            file_names = [' combined param correct ', ' combined opto correct ', 
                          ' combined param response ', ' combined opto response ']
            
        else:
            param=='opto'
            plot_param(d, param=param, showTrialN=True, ignoreNoRespAfter=None, returnArray=False)
            
file_names = [' combined param correct ', ' combined opto correct ', 
                          ' combined param response ', ' combined opto response ']

        for f in file_names:

            plt.savefig(path + '/' + mouse_id + f + date + '.png', dpi=300)
            plt.close()
            
        
#        onset = d['optoOnset'][:]
#        onset = np.insert(onset,0, -1)
#        for i, val in enumerate(np.flip(onset)):
#            if val==-1:
#                val='no opto'
#            plt.savefig(path +  '/' + mouse_id +
#                        ' ' + param + ' ' + str(val) + ' opto response rate ' + date + '.png', dpi=300)
#            plt.close()
#            
#            plt.savefig(path +  '/' + mouse_id + 
#                        ' ' + param + ' ' + str(val) + ' opto correct given response ' + date + '.png', dpi=300)
#            plt.close()
        
        
