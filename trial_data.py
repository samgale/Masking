# -*- coding: utf-8 -*-
"""
Created on Wed May 01 10:47:39 2019

@author: svc_ccg
"""

from __future__ import division
import os
import numpy as np
import h5py
import datetime 
from matplotlib import pyplot as plt
import pandas as pd

# for each mouse
def get_files(mouse_id):
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'files_to_analyze')
    files = os.listdir(dataDir)
    files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[2],'%Y%m%d'))
    return [os.path.join(dataDir,f) for f in files]  
    
              
def trials(data):
    trialResponse = d['trialResponse'].value
    answered_trials = np.count_nonzero(trialResponse)
    correct = (trialResponse==1).sum()
    percentCorrect = (correct/float(answered_trials)) * 100
    return percentCorrect



mice = ['439508', '439506', '439502', '441357', '441358']

df = pd.DataFrame()

for mouse in mice:
    files = get_files(mouse)
    print(mouse)
    for i, f in enumerate(files):
        d = h5py.File(f)
        #df.append((trials(d)))
        print(trials(d))
        

        
      
