# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:56:20 2019

@author: chelsea.strawder

Returns a list of trials indices where they moved the wheel before a threshold
Use these indices to exclude those trials from analysis
rxnTimes(d) returns an np.array with 3 lists, the 3rd being ignoreTrials 

"""

from dataAnalysis import rxnTimes


def ignore_trials(data):
    d = data
    return rxnTimes(d)[2]



