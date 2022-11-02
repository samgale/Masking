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
    
    # signals,t,dt = getInputSignals()

    trialsPerCondition = 500
    targetSide = (1,) # (1,0) (-1,1,0)
    optoOnset = [np.nan]
    optoSide = [0]

    # mice
    maskOnset = [0,2,3,4,6,np.nan]
    trialEnd = 60
    
    respRateData = np.load(os.path.join(maskDataPath,'respRate_mice.npz'))
    respRateMean = respRateData['mean'][:-1]
    respRateSem = respRateData['sem'][:-1]
    
    fracCorrData = np.load(os.path.join(maskDataPath,'fracCorr_mice.npz'))
    fracCorrMean = fracCorrData['mean'][:-1]
    fracCorrSem = fracCorrData['sem'][:-1]
    
    reacTimeData = np.load(os.path.join(maskDataPath,'reacTime_mice.npz'))
    reacTimeMean = reacTimeData['mean'][:-1] / dt
    reacTimeSem = reacTimeData['sem'][:-1] / dt
    
    toUse = [True,True,False,True,True,True]
    maskOnset = list(np.array(maskOnset)[toUse])
    respRateMean = respRateMean[toUse]
    respRateSem = respRateSem[toUse]
    fracCorrMean = fracCorrMean[toUse]
    fracCorrSem = fracCorrSem[toUse]
    reacTimeMean = reacTimeMean[toUse]
    reacTimeSem = reacTimeSem[toUse]

    # humans
    # maskOnset = [0,2,4,6,8,10,12,np.nan]
    # trialEnd = 240
    
    # respRateData = np.load(os.path.join(maskDataPath,'respRate_humans.npz'))
    # respRateMean = respRateData['mean'][:-1]
    # respRateSem = respRateData['sem'][:-1]
    
    # fracCorrData = np.load(os.path.join(maskDataPath,'fracCorr_humans.npz'))
    # fracCorrMean = fracCorrData['mean'][:-1]
    # fracCorrSem = fracCorrData['sem'][:-1]
    
    # reacTimeData = np.load(os.path.join(maskDataPath,'reacTime_humans.npz'))
    # reacTimeMean = reacTimeData['mean'][:-1] / dt
    # reacTimeSem = reacTimeData['sem'][:-1] / dt
    
    # toUse = [True,True,True,True,False,False,False,True]
    # maskOnset = list(np.array(maskOnset)[toUse])
    # respRateMean = respRateMean[toUse]
    # respRateSem = respRateSem[toUse]
    # fracCorrMean = fracCorrMean[toUse]
    # fracCorrSem = fracCorrSem[toUse]
    # reacTimeMean = reacTimeMean[toUse]
    # reacTimeSem = reacTimeSem[toUse]

    
    fixedParams = (signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,respRateSem,fracCorrMean,fracCorrSem,reacTimeMean,reacTimeSem)

    tauIRange = np.arange(0.5,4,0.5)
    alphaRange = np.arange(0,0.5,0.05)
    etaRange = [1]
    sigmaRange = np.arange(0.4,1.6,0.1)
    tauARange = np.arange(1,10,1)
    decayRange = np.arange(0,1.1,0.1)
    inhibRange = np.arange(0,1.1,0.1)
    thresholdRange = np.arange(0.8,5.2,0.2)
    trialEndRange = [trialEnd]
    postDecisionRange = np.arange(4,24,2)


    fitParamRanges = (tauIRange,alphaRange,etaRange,sigmaRange,tauARange,decayRange,inhibRange,thresholdRange,trialEndRange,postDecisionRange)   
    
    fitParamsIter = itertools.product(*fitParamRanges)
    
    nParamCombos = np.prod([len(p) for p in fitParamRanges])
    paramCombosPerJob = int(nParamCombos/totalJobs)
    paramsStart = jobInd * paramCombosPerJob

    bestFitParams = None
    bestFitError = None
    for fitParams in itertools.islice(fitParamsIter,paramsStart,paramsStart+paramCombosPerJob):
        modelError = calcModelError(fitParams,*fixedParams)
        if bestFitError is None or modelError < bestFitError:
            bestFitParams = fitParams
            bestFitError = modelError
    np.savez(os.path.join(baseDir,'HPC','fit_'+str(jobInd)+'.npz'),params=bestFitParams,error=bestFitError)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobInd',type=int)
    parser.add_argument('--totalJobs',type=int)
    args = parser.parse_args()
    findBestFit(args.jobInd,args.totalJobs)
