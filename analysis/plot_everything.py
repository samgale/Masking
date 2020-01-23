# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:57:35 2020

@author: chelsea.strawder
"""

import h5py
import fileIO
import behaviorAnalysis
import performanceBySOA
from plottingTargetContrast import plot_contrast
from plotting_target_length import plot_flash
from SessionPerformance import plot_session


# choose mouse file
f = fileIO.getFile(rootDir=r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking')
d = h5py.File(f)


# plot session wheel trace - 1 plot, unless mask==True - 2 plots
behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], 
                               ignoreRepeats=True, framesToShowBeforeStart=30, mask=False, maskOnly=False)


# plot target duration responses - 3 plots
plot_flash(d)


# plot contrast responses - 3 plots
plot_contrast(d)


# plot masking session - 3 plots 
performanceBySOA.plot_soa(d)


# plot activity over entire session, trial-by-trial - 1 plot
plot_session(d)