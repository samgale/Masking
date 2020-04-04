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
import dataAnalysis as da

# choose mouse file
d = da.import_data()

# plot session wheel trace - 1 plot, unless mask==True - 2 plots
##  if session is from 1/13 - 1/28, use framesToShowBeforeStart=30, else 60
behaviorAnalysis.makeWheelPlot(d, responseFilter=[0], 
                               ignoreRepeats=True, framesToShowBeforeStart=60, 
                               mask=True, maskOnly=True)


# plot target duration responses - 3 plots
plot_flash(d)


# plot contrast responses - 3 plots
plot_contrast(d)


# plot masking session - 3 plots 
performanceBySOA.plot_soa(d)


# plot activity over entire session, trial-by-trial - 1 plot
plot_session(d)


# plot reaction time by parameter - 1 plot
# (df, selection='all', param='soa', stat='Median', errorBars=False)
plot_by_param(da.create_df(d))


# plot multiple masking session response times
files = behaviorAnalysis.get_files('486634','masking_to_analyze')    #imports ALL masking files for mouse
dn = da.combine_files(files, '212','213','214')
dfall = da.combine_dfs(dn)
plot_by_param(dfall, 'soa')
