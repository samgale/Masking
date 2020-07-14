# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:59:57 2020

@author: chelsea.strawder
"""

import behaviorAnalysis
import performanceBySOA
from plottingTargetContrast import plot_contrast
from plotting_target_lengths import plot_flash
from SessionPerformance import plot_session
from responsePlotByParam import plot_by_param
import qualityControl
from catchTrials import catch_trials
import matplotlib.pyplot as plt
import os


def save_daily_plots(d):

    mouse_id=d['subjectName'][()]
    date = d['startTime'][()].split('_')[0][-4:]
    date = date[:2]+'-'+date[2:]
    
    date = date if date[:2] in ['10','11','12'] else date[-4:]
    
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking\active_mice'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'Plots') 
    
    
    wheelDir = os.path.join(dataDir, 'Wheel plots')
    
# daily wheel plot
    behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], 
                                   ignoreRepeats=True, framesToShowBeforeStart=0, 
                                   mask=False, maskOnly=False, xlim=[0, .8], ylim='auto')
    
    
    plt.savefig(wheelDir+'/Daily Wheel/' + mouse_id + ' ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plot no response trials only (with repeats)
    behaviorAnalysis.makeWheelPlot(d, responseFilter=[0], 
                                   ignoreRepeats=False, framesToShowBeforeStart=0, 
                                   mask=False, maskOnly=False,  xlim=[0, .8], ylim=[-10,10])
    
    plt.rcParams.update({'font.size': 10})
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.96)
    
    plt.savefig(wheelDir+'/No Resp Wheel/' + mouse_id + ' ' + date + ' no resp.png', dpi=300, bbox_inches='tight')
    plt.close()
    
# plots catch trial wheel traces 
    catch_trials(d, xlim=[0, .8]) 
    
    plt.savefig(wheelDir+'/Catch/' + mouse_id + ' catch ' + date + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
# plot activity over entire session, trial-by-trial - 1 plot
    plot_session(d)
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




