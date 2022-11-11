# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:27 2021

@author: svc_ccg
"""

import copy
import pickle
import random
import numpy as np
import scipy.optimize
from numba import njit


def getInputSignals(psthFilePath=None):
    signalNames = ('targetOnly','maskOnly','mask')
    dt = 1/120*1000
    trialEndTimeMax = 2500
    trialEndMax = int(round(trialEndTimeMax/dt))
    t = np.arange(0,trialEndMax*dt+dt,dt)
    
    if psthFilePath is not None:
        popPsth = pickle.load(open(psthFilePath,'rb'))
        popPsthIntp = {}
        for sig in signalNames:
            popPsthIntp[sig] = {}
            for hemi in ('ipsi','contra'):
                popPsthIntp[sig][hemi] = {}
                for mo in popPsth[sig][hemi]:
                    p = popPsth[sig][hemi][mo].copy()
                    p -= p[:,popPsth['t']<0].mean(axis=1)[:,None]
                    p = np.nanmean(p,axis=0)
                    p[popPsth['t']>0.2] = 0
                    p = np.interp(t,popPsth['t']*1000,p)
                    
                    # p -= p[t<30].mean()
                    # p[0] = 0
                    
                    p[t<30] = 0
                    p[(t<30) | (t>200)] = 0
                    p[p<0] = 0
                    if hemi=='ipsi':
                        if sig=='targetOnly':
                            p[:] = 0
                        else:
                            p[t<mo*dt+30] = 0
                    
                    maskOn = np.nan if sig=='targetOnly' else mo
                    popPsthIntp[sig][hemi][maskOn] = p                                     
        signals = copy.deepcopy(popPsthIntp)
        smax = max([signals[sig][hemi][mo].max() for sig in signals.keys() for hemi in ('ipsi','contra') for mo in signals[sig][hemi]])
        for sig in signals.keys():
            for hemi in ('ipsi','contra'):
                for mo in signals[sig][hemi]:
                    s = signals[sig][hemi][mo]
                    s /= smax                                    
    else:
        latency = 4
        targetDur = 6
        maskDur = 6
        signals = {}
        maskOnset = [0,2,3,4,6,8,10,12,np.nan]
        for sig in signalNames:
            signals[sig] = {}
            for hemi in ('ipsi','contra'):
                signals[sig][hemi] = {}
                for mo in maskOnset:
                    if (sig=='targetOnly' and not np.isnan(mo)) or (sig=='maskOnly' and mo!=0) or (sig=='mask' and not mo>0):
                        continue
                    s = np.zeros(t.size)
                    if sig in ('targetOnly','mask') and hemi=='contra':
                        s[latency:latency+targetDur] = 1
                    if 'mask' in sig:
                        s[latency+mo:latency+mo+maskDur] = 1
                    signals[sig][hemi][mo] = s
    return signals,t,dt


def fitModel(fitParamRanges,fixedParams,finish=False):
    fit = scipy.optimize.brute(calcModelError,fitParamRanges,args=fixedParams,full_output=False,finish=None,workers=1)
    if finish:
        finishRanges = []
        for rng,val in zip(fitParamRanges,fit):
            if val in (rng.start,rng.stop):
                finishRanges.append(slice(val,val+1,1))
            else:
                oldStep = rng.step
                newStep = oldStep/4
                finishRanges.append(slice(val-oldStep+newStep,val+oldStep,newStep))
        fit = scipy.optimize.brute(calcModelError,finishRanges,args=fixedParams,full_output=False,finish=None,workers=1)
    return fit


def calcModelError(paramsToFit,*fixedParams):
    tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision = paramsToFit
    signals,targetSide,maskOnset,optoOnset,optoSide,trialsPerCondition,respRateMean,respRateSem,fracCorrMean,fracCorrSem,reacTimeMean,reacTimeSem,reacTimeCorrectMean,reacTimeCorrectSem,reacTimeIncorrectMean,reacTimeIncorrectSem = fixedParams
    sessionData = runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,trialsPerCondition)
    if sessionData is None:
        return 1e6
    else:
        trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord,Rrecord = sessionData
        result = analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime)
        respRateError = np.nansum(((respRateMean-result['responseRate'])/respRateSem)**2)
        fracCorrError = np.nansum(((fracCorrMean-result['fractionCorrect'])/fracCorrSem)**2)
        # respRateError = np.nansum((respRateMean-result['responseRate'])**2)
        # fracCorrError = np.nansum((2*(fracCorrMean-result['fractionCorrect']))**2)
        if postDecision > 0:
            # respTimeError = np.nansum(((reacTimeMean-(result['responseTimeMedian']+postDecision))/reacTimeSem)**2)
            respTimeCorrectError = np.nansum(((reacTimeCorrectMean-(result['responseTimeCorrectMedian']+postDecision))/reacTimeCorrectSem)**2)
            respTimeIncorrectError = np.nansum(((reacTimeIncorrectMean-(result['responseTimeIncorrectMedian']+postDecision))/reacTimeIncorrectSem)**2)
            # respTimeError = np.nansum(((reacTimeMean-(result['responseTimeMedian']+postDecision))/(np.nanmax(reacTimeMean)-np.nanmin(reacTimeMean)))**2)
        else:
            # respTimeError = 0
            respTimeCorrectError = respTimeIncorrectError = 0
        return respRateError + fracCorrError + (respTimeCorrectError + respTimeIncorrectError)/2


def analyzeSession(targetSide,maskOnset,optoOnset,optoSide,trialTargetSide,trialMaskOnset,trialOptoOnset,trialOptoSide,response,responseTime,Lrecord=None,Rrecord=None):
    result = {}
    responseRate = []
    fractionCorrect = []
    responseTimeMedian = []
    responseTimeCorrectMedian = []
    responseTimeIncorrectMedian = []
    for side in targetSide:
        result[side] = {}
        sideTrials = trialTargetSide==side
        mo = [np.nan] if side==0 else maskOnset
        for maskOn in mo:
            result[side][maskOn] = {}
            maskTrials = np.isnan(trialMaskOnset) if np.isnan(maskOn) else trialMaskOnset==maskOn
            for optoOn in optoOnset:
                result[side][maskOn][optoOn] = {}
                for opSide in optoSide:
                    optoTrials = np.isnan(trialOptoOnset) if np.isnan(optoOn) else (trialOptoOnset==optoOn) & (trialOptoSide==opSide)
                    trials = sideTrials & maskTrials & optoTrials
                    responded = response[trials]!=0
                    responseRate.append(np.sum(responded)/np.sum(trials))
                    responseTimeMedian.append(np.median(responseTime[trials][responded]))
                    result[side][maskOn][optoOn][opSide] = {}
                    result[side][maskOn][optoOn][opSide]['responseRate'] = responseRate[-1]
                    result[side][maskOn][optoOn][opSide]['responseTime'] = responseTime[trials][responded]
                    if side!=0 and maskOn!=0:
                        correct = response[trials]==side
                        fractionCorrect.append(np.sum(correct[responded])/np.sum(responded))
                        responseTimeCorrectMedian.append(np.median(responseTime[trials][responded & correct]))
                        responseTimeIncorrectMedian.append(np.median(responseTime[trials][responded & (~correct)]))
                        result[side][maskOn][optoOn][opSide]['fractionCorrect'] = fractionCorrect[-1]
                        result[side][maskOn][optoOn][opSide]['responseTimeCorrect'] = responseTime[trials][responded & correct]
                        result[side][maskOn][optoOn][opSide]['responseTimeIncorrect'] = responseTime[trials][responded & (~correct)]
                    else:
                        fractionCorrect.append(np.nan)
                        responseTimeCorrectMedian.append(np.nan)
                        responseTimeIncorrectMedian.append(np.nan)
    result['responseRate'] = np.array(responseRate)
    result['fractionCorrect'] = np.array(fractionCorrect)
    result['responseTimeMedian'] = np.array(responseTimeMedian)
    result['responseTimeCorrectMedian'] = np.array(responseTimeCorrectMedian)
    result['responseTimeIncorrectMedian'] = np.array(responseTimeIncorrectMedian)
    return result


def runSession(signals,targetSide,maskOnset,optoOnset,optoSide,tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,trialsPerCondition,optoLatency=0,record=False):
    trialTargetSide = []
    trialMaskOnset = []
    trialOptoOnset = []
    trialOptoSide = []
    response = []
    responseTime = []
    if record:
        Lrecord = []
        Rrecord = []
    else:
        Lrecord = Rrecord = None
    for side in targetSide:
        mo = [np.nan] if side==0 else maskOnset
        for maskOn in mo:
            if np.isnan(maskOn):
                sig = 'targetOnly'
                maskOn = np.nan
            elif maskOn==0:
                sig = 'maskOnly'
            else:
                sig = 'mask'
            for optoOn in optoOnset:
                for opSide in optoSide:
                    if side==0:
                        Lsignal = np.zeros(signals[sig]['ipsi'][maskOn].size)
                        Rsignal = Lsignal.copy()
                    elif side<0:
                        Lsignal = signals[sig]['contra'][maskOn].copy()
                        Rsignal = signals[sig]['ipsi'][maskOn].copy()
                    else:
                        Lsignal = signals[sig]['ipsi'][maskOn].copy()
                        Rsignal = signals[sig]['contra'][maskOn].copy()
                    if not np.isnan(optoOn):
                        i = int(optoOn+optoLatency)
                        if opSide <= 0:
                            Lsignal[i:] = 0
                        if opSide >= 0:
                            Rsignal[i:] = 0
                    if tauI==0 and alpha > 0:
                        for s in (Lsignal,Rsignal):
                            i = s > 0
                            s[i] = s[i]**eta / (alpha**eta + s[i]**eta)
                            s *= alpha**eta + 1
                    for n in range(trialsPerCondition):
                        trialTargetSide.append(side)
                        trialMaskOnset.append(maskOn)
                        trialOptoOnset.append(optoOn)
                        trialOptoSide.append(opSide)
                        result = runTrial(tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,Lsignal,Rsignal,record)
                        response.append(result[0])
                        responseTime.append(result[1])
                        if record:
                            Lrecord.append(result[2])
                            Rrecord.append(result[3])
                        if n > 0.1 * trialsPerCondition and not any(response):
                            return
    return np.array(trialTargetSide),np.array(trialMaskOnset),np.array(trialOptoOnset),np.array(trialOptoSide),np.array(response),np.array(responseTime),Lrecord,Rrecord


@njit
def runTrial(tauI,alpha,eta,sigma,tauA,decay,inhib,threshold,trialEnd,postDecision,Lsignal,Rsignal,record=False):
    if record:
        Lrecord = np.full(int(trialEnd),np.nan)
        Rrecord = Lrecord.copy()
    else:
        Lrecord = Rrecord = None
    L = R = 0
    iL = iR = 0
    t = 0
    response = 0
    while t < trialEnd - postDecision and response == 0:
        if record:
            Lrecord[t] = L
            Rrecord[t] = R
        if L > threshold and R > threshold:
            response = -1 if L > R else 1
        elif L > threshold:
            response = -1
        elif R > threshold:
            response = 1
        if t >= Lsignal.size:
            Lsig = Rsig = 0
        elif alpha > 0:
            Lsig = (Lsignal[t]**eta) / (alpha**eta + iL**eta) if Lsignal[t]>0 and iL>=0 else Lsignal[t]
            Rsig = (Rsignal[t]**eta) / (alpha**eta + iR**eta) if Rsignal[t]>0 and iR>=0 else Rsignal[t]
        else:
            Lsig = Lsignal[t]
            Rsig = Rsignal[t]
        Lnow = L
        L += (random.gauss(0,sigma) + Lsig - decay*L - inhib*R) / tauA
        R += (random.gauss(0,sigma) + Rsig - decay*R - inhib*Lnow) / tauA
        if tauI > 0:
            if t >= Lsignal.size:
                iL -= iL / tauI
                iR -= iR / tauI
            else:
                iL += (Lsignal[t] - iL) / tauI
                iR += (Rsignal[t] - iR) / tauI
        t += 1
    responseTime = t-1
    return response,responseTime,Lrecord,Rrecord

