# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
from simple_slurm import Slurm


# script to run
script_path = os.path.join(os.path.expanduser('~'),'PythonScripts','maskTaskModelHPC.py')
print(f'running {script_path}')

# define the job record output folder
stdout_location = os.path.join(os.path.expanduser('~'),'job_records')
# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

# build the python path
conda_environment = 'maskTaskModel'
python_path = os.path.join(os.path.expanduser('~'), 
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

# call the `sbatch` command to run the jobs
totalJobs = 1000
for jobInd in range(totalJobs):
    slurm.sbatch('{} {} --jobInd {} --totalJobs {}'.format(
                 python_path,
                 script_path,
                 jobInd,
                 totalJobs))
