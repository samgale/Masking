# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:56:13 2020

pulls 3 most recent masking session files and creates dataframes for each day
then combines the data frames and analyzes the combined data frames
plots avg performance (reaction time, response time, response latency)

@author: svc_ccg
"""

import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from dataAnalysis import create_df
from behaviorAnalysis import get_files, formatFigure
from responsePlotByParam import plot_by_param

mouse='495786'
files = get_files(mouse,'masking_to_analyze')    #imports all masking files for mouse

dn = {}
for i, f in enumerate(files[:-3]):
    d = h5py.File(f) 
    dn['df_{}'.format(i)] = create_df(d)        #creates keys in dict dn with format df_#

dictget = lambda x, *k: [x[i] for i in k]
df1, df2, df3 = dictget(dn, 'df_0', 'df_1', 'df_2')    # assigns each dataframe to a new variable - essentially unpacks dict dn

dfall = pd.concat([df1,df2,df3], ignore_index=True)

plot_by_param('soa', dfall)
    
    
    