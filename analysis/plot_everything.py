# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:57:35 2020

@author: chelsea.strawder
"""


import behaviorAnalysis
import performanceBySOA
from plottingTargetContrast import plot_contrast
from plotting_target_lengths import plot_flash
from SessionPerformance import plot_session
from responsePlotByParam import plot_by_param
import dataAnalysis
import qualityControl
from percentCorrect import session_stats
from catchTrials import catch_trials


# choose mouse file
d = dataAnalysis.import_data()


# prints out performance counts/% from session
session_stats(d, returnAs = 'print')


# plot session wheel trace - 1 plot, unless mask==True - 2 plots
##  if session is from 1/13 - 1/28, use framesToShowBeforeStart=30, else 60

behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], 
                               ignoreRepeats=True, framesToShowBeforeStart=0, 
                               mask=False, maskOnly=False, xlim=[0, .8], ylim=[-20,20])

# plot no response trials only (with repeats)
behaviorAnalysis.makeWheelPlot(d, responseFilter=[0], 
                               ignoreRepeats=False, framesToShowBeforeStart=0, 
                               mask=False, maskOnly=False,  xlim=[0, .8], ylim=[-10,10])


# plots catch trial wheel traces 
catch_trials(d, xlim=[0,.8], ylim='auto', plot_ignore=True)   



# plot activity over entire session, trial-by-trial - 1 plot
plot_session(d)


# check for dropped frames
qualityControl.check_frame_intervals(d)


# check number of quiescent period violations - use 'sum' for cumsum OR 'count' for count per trial
qualityControl.check_qviolations(d, plot_type='sum')

qualityControl.check_qviolations(d, plot_type='count')


# check distribution of delta wheel position 
qualityControl.check_wheel(d)

d.close()


# plot target duration responses - 3 plots ###################################
plot_flash(d)


# plot contrast responses - 3 plots
plot_contrast(d)


# plot masking session - 3 plots 
performanceBySOA.plot_soa(d)


# plot reaction time by parameter - 1 plot
# (df, selection='all', param1='soa', param2='trialLength', stat='Median', errorBars=False)
plot_by_param(dataAnalysis.create_df(d), param2='outcomeTime')


# plot multiple masking session response times
files = behaviorAnalysis.get_files('486634','masking_to_analyze')    #imports ALL masking files for mouse
dn = dataAnalysis.combine_files(files, '212','213','214')
dfall = dataAnalysis.combine_dfs(dn)
plot_by_param(dfall, 'soa')
