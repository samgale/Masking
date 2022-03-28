# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:56:13 2020

pulls 3 most recent masking session files and creates dataframes for each day
then combines the data frames and analyzes the combined data frames
plots avg performance (reaction time)

@author: svc_ccg
"""

from dataAnalysis import combine_files, combine_dfs
from behaviorAnalysis import get_files
from responsePlotByParam import plot_by_param

files = get_files('486634','masking_to_analyze')    #imports ALL masking files for mouse

dn = combine_files(files, '212','213','214')

dfall = combine_dfs(dn)

plot_by_param(dfall, 'soa')
    
    