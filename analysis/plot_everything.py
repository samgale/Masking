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
from QualityControl import check_frame_intervals
from percentCorrect import session_stats
from catchTrials import catch_trials


# choose mouse file
d = dataAnalysis.import_data()

# plot session wheel trace - 1 plot, unless mask==True - 2 plots
##  if session is from 1/13 - 1/28, use framesToShowBeforeStart=30, else 60

behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], 
                               ignoreRepeats=True, framesToShowBeforeStart=60, 
                               mask=False, maskOnly=False)

# plot activity over entire session, trial-by-trial - 1 plot
plot_session(d)


# plots catch trial wheel traces 
catch_trials(d)


# prints out performance counts/% from session
session_stats(d)


# check for dropped frames
check_frame_intervals(d)


# plot target duration responses - 3 plots
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
