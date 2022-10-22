# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import itertools
import os
import numpy as np
from maskTaskModelUtils import getInputSignals,calcModelError


baseDir = '/allen/programs/braintv/workgroups/tiny-blue-dot/masking/Sam'

def findBestFit(jobInd,totalJobs):
    
    maskDataPath = os.path.join(baseDir,'Analysis')

    signals,t,dt = getInputSignals(psthFilePath=os.path.join(maskDataPath,'popPsth.pkl'))

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
    
    fixedParams = (signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,fracCorrMean,reacTimeMean)

    tauIRange = np.arange(0.5,3,0.5)
    alphaRange = np.arange(0.05,0.3,0.05)
    etaRange = [1]
    sigmaRange = np.arange(0.1,1.5,0.1)
    tauARange = np.arange(1,10,1)
    decayRange = np.arange(0,1.1,0.1)
    inhibRange = np.arange(0,1.1,0.1)
    thresholdRange = np.arange(0.5,2.2,0.2)
    trialEndRange = [trialEnd]
    postDecisionRange = np.arange(6,30,3)

    fitParamRanges = (tauIRange,alphaRange,etaRange,sigmaRange,tauARange,decayRange,inhibRange,thresholdRange,trialEndRange,postDecisionRange)   
    
    fitParamsIter = itertools.product(*fitParamRanges)
    
    nParamCombos = np.prod([len(p) for p in fitParamRanges])
    paramCombosPerJob = int(nParamCombos/totalJobs)
    paramsStart = jobInd * paramCombosPerJob

    bestFitParams = None
    bestFitError = 1e6
    for fitParams in itertools.islice(fitParamsIter,paramsStart,paramsStart+paramCombosPerJob):
        modelError = calcModelError(fitParams,*fixedParams)
        if modelError < bestFitError:
            bestFitParams = fitParams
            bestFitError = modelError
    np.savez(os.path.join(baseDir,'HPC','fit_'+str(jobInd)+'.npz'),params=bestFitParams,error=bestFitError)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobInd',type=int)
    parser.add_argument('--totalJobs',type=int)
    args = parser.parse_args()
    findBestFit(args.jobInd,args.totalJobs)
