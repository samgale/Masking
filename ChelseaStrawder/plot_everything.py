# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:57:35 2020

@author: chelsea.strawder
"""


import behaviorAnalysis
from plotting_variable_params import plot_param
from SessionPerformance import plot_session
from responsePlotByParam import plot_by_param
import dataAnalysis
import qualityControl
from percentCorrect import session_stats
from catchTrials import catch_trials
import matplotlib.pyplot as plt
from save_plots import save_daily_plots
from create_pdf import create_daily_summary
from dataAnalysis import import_data



maxConsecutiveNoResp = 10
d = import_data()
save_daily_plots(d, maxConsecutiveNoResp)  
create_daily_summary(d, maxConsecutiveNoResp)
d.close()


# choose mouse file
d = dataAnalysis.import_data()

plt.ion()

# prints out performance counts/% from session
session_stats(d, returnAs = 'print')


# plot session wheel trace - 1 plot, unless mask==True - 2 plots
##  if session is from 1/13 - 1/28, use framesToShowBeforeStart=30, else 60d

behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], ignoreRepeats=True, 
                               framesToShowBeforeStart=0, mask=False, maskOnly=False, 
                               xlim=[0, .8], ylim='auto', ignoreNoRespAfter=10, use_legend=True )

# plot no response trials only (with repeats)
behaviorAnalysis.makeWheelPlot(d, responseFilter=[0], ignoreRepeats=False, 
                               framesToShowBeforeStart=0, mask=False, maskOnly=False,  
                               xlim=[0, .8], ylim=[-8,8], ignoreNoRespAfter=10 )


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
plot_param(d, 'targetFrames')


# plot contrast responses - 3 plots
plot_param(d, 'targetContrast')


# plot masking session - 3 plots 
plot_param(d, 'soa')

# plot reaction time by parameter - 1 plot
# (df, selection='all', param1='soa', param2='trialLength', stat='Median', errorBars=False)
plot_by_param(dataAnalysis.create_df(d), param2='outcomeTime')


# plot multiple masking session response times
files = behaviorAnalysis.get_files('486634','masking_to_analyze')    #imports ALL masking files for mouse
dn = dataAnalysis.combine_files(files, '212','213','214')
dfall = dataAnalysis.combine_dfs(dn)
plot_by_param(dfall, 'soa')
