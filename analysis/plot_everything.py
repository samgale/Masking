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


# plot session wheel trace
behaviorAnalysis.makeWheelPlot(d, responseFilter=[-1,0,1], 
                               ignoreRepeats=True, framesToShowBeforeStart=30, mask=False, maskOnly=False)


# plot target duration session
plot_flash(d)


# plot contrast session
plot_contrast(d)


# plot mask sessions
performanceBySOA.plot_soa(d)


# plot activity over entire session, trial-by-trial
plot_session(d)