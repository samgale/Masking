# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import os
import argparse
import numpy as np
from maskTaskModelUtils import calcModelError

outputPath = r'/allen/programs/braintv/workgroups/tiny-blue-dot/masking/Sam/HPC'

def saveBestFit(n,fixedParams,fitParamsIter):
    bestFitParams = None
    bestFitError = 1e6
    for fitParams in fitParamsIter:
        modelError = calcModelError(fitParams,fixedParams)
        if modelError < bestFitError:
            bestFitParams = fitParams
            bestFitError = modelError
    np.savez(os.path.join(outputPath,'fit_'+str(n)+'.npz'),params=bestFitParams,error=bestFitError)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int)
    parser.add_argument('--fixedParams',type=list)
    parser.add_argument('--fitParams',type=list)
    args = parser.parse_args()
    saveBestFit(args.n,args.fixedParams,args.fitParams)
