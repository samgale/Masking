# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:56:20 2019

@author: chelsea.strawder

Returns a list of trials indices where they moved the wheel before a threshold
Use these indices to exclude those trials from analysis
rxnTimes(d) returns an np.array with 3 lists, the 3rd being ignoreTrials 

"""



def ignore_trials(data):
    from dataAnalysis import rxnTimes, create_df

    df = create_df(data)
    return rxnTimes(data, df)[2]



