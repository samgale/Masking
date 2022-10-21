# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import itertools
import pickle
import numpy as np
from simple_slurm import Slurm




# script to run
script_path = '~/PythonScripts/maskTaskModelHPC.py'
print(f'running {script_path}')

# define the job record output folder
stdout_location = os.path.join(os.path.expanduser("~"), 'job_records')
# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

# build the python path
conda_environment = 'maskTaskModel'
python_path = os.path.join(os.path.expanduser("~"), 
                           'miniconda3', 
                           'envs', 
                           conda_environment,
                           'bin',
                           'python')

slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='maskTaskModel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              # output=f'{stdout_location}/log.out',
              #mail_type=["FAIL"],          # Mail events (NONE, BEGIN, END, FAIL, ALL)
              #mail_user="corbettb@alleninstitute.org",     # Where to send mail  
              time='99:00:00',
              mem_per_cpu='16gb')

# split fit parameter sets into jobs
maskDataPath = r"\\allen\programs\braintv\workgroups\tiny-blue-dot\masking\Sam\Analysis"

signals = popPsth = pickle.load(open(os.path.join(maskDataPath,'modelInputSignals.pkl'),'rb'))
dt = 1/120.0*1000

trialsPerCondition = 500
targetSide = (1,) # (1,0) (-1,1,0)
optoOnset = [np.nan]
optoSide = [0]

# mice
maskOnset = [0,2,3,4,6,np.nan]
trialEnd = 78

respRateData = np.load(os.path.join(maskDataPath,'respRate_mice.npz'))
respRateMean = respRateData['mean'][:-1]

fracCorrData = np.load(os.path.join(maskDataPath,'fracCorr_mice.npz'))
fracCorrMean = fracCorrData['mean'][:-1]

reacTimeData = np.load(os.path.join(maskDataPath,'reacTime_mice.npz'))
reacTimeMean = reacTimeData['mean'][:-1] / dt

# humans
# maskOnset = [0,2,4,6,8,10,12,np.nan]
# trialEnd = 300

# respRateData = np.load(os.path.join(maskDataPath,'respRate_humans.npz'))
# respRateMean = respRateData['mean'][:-1]

# fracCorrData = np.load(os.path.join(maskDataPath,'fracCorr_humans.npz'))
# fracCorrMean = fracCorrData['mean'][:-1]

# reacTimeData = np.load(os.path.join(maskDataPath,'reacTime_humans.npz'))
# reacTimeMean = reacTimeData['mean'][:-1] / dt


tauIRange = np.arange(0.3,1.2,0.2)
alphaRange = np.arange(0.05,0.2,0.05)
etaRange = [1]
sigmaRange = np.arange(0.2,1.2,0.1)
tauARange = np.arange(1,10,1)
decayRange = np.arange(0,1.1,0.2)
inhibRange = np.arange(0,1,0.1)
thresholdRange = np.arange(0.6,2,0.1)
trialEndRange = [trialEnd]
postDecisionRange = np.arange(6,30,6)

params = (tauIRange,alphaRange,etaRange,sigmaRange,tauARange,decayRange,inhibRange,thresholdRange,trialEndRange,postDecisionRange)

nParamCombos = np.prod([len(p) for p in params])

fixedParams = (signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,fracCorrMean,reacTimeMean)


# call the `sbatch` command to run the jobs
nJobs = 1000
paramCombosPerJob = int(nParamCombos/nJobs)
for n,i in enumerate(range(0,nParamCombos,paramCombosPerJob)):
    fitParamsIter = itertools.product(*params)
    slurm.sbatch('{} {} --n {} --fixedParams {} --fitParamsIter {}'.format(
            python_path,
            script_path,
            n,
            fixedParams,
            itertools.islice(fitParamsIter,i,i+paramCombosPerJob)))
    
    