# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:45:29 2020

@author: chelsea.strawder

I'm still working on this plotting 
Creates plots of trial length by SOA (or specified parameter)

can easily call using plot_by_param(create_df(import_data()))
"""


import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

def plot_by_param(param, df):    #param = soa, targetContrast, or targetLength

    matplotlib.rcParams['pdf.fonttype'] = 42
    sns.set_style('white')

    nonzeroRxns = df[(df['trialLength']!=df['trialLength'].max()) & 
                     (df['ignoreTrial']!=True) & (df['resp']!=0)]
    
    corrNonzero = nonzeroRxns[(nonzeroRxns['resp']==1) & (nonzeroRxns['nogo']==False)]
    missNonzero = nonzeroRxns[(nonzeroRxns['resp']==-1) & (nonzeroRxns['nogo']==False)]
    
    miss = missNonzero.pivot_table(values='trialLength', index=param, columns='rewDir', 
                            margins=True, dropna=True)
    hit = corrNonzero.pivot_table(values='trialLength', index=param, columns='rewDir', 
                            margins=True, dropna=True)
    
    #hit.plot(title='hits')
    #miss.plot(title='misses')
    print('hits \n', hit)
    print('\n' * 2)
    print('misses \n', miss)



    
    # use the df to filter the trial by RewDir 
        # maybe use multiindex?? 
    
    y = corrNonzero.groupby(['rewDir', param])['trialLength'].mean()
    
    #to reduce bulk below; something like this?
    Rhit = corrNonzero[corrNonzero['rewDir']==1]
    avgs = Rhit.groupby('soa')['trialLength'].mean()
        

 ### how to make this less bulky/redundant??     
    
    hits = [[],[]]  #R, L
    misses = [[],[]]
    maskOnly = []
    
    for val in np.unique(df[param]):
        hitVal = [[],[]]
        missVal = [[],[]]
        for j, (time, p, resp, direc, mask) in enumerate(zip(
                nonzeroRxns['trialLength'], nonzeroRxns[param], nonzeroRxns['resp'], 
                nonzeroRxns['rewDir'], nonzeroRxns['maskContrast'])):
            if p==val:  
                if direc==1:       # soa=0 is targetOnly, R turning
                    if resp==1:
                        hitVal[0].append(time)  
                    else:
                        missVal[0].append(time)  
                elif direc==-1:   # soa=0 is targetOnly, L turning
                    if resp==1:
                        hitVal[1].append(time)  
                    else:
                        missVal[1].append(time)
                elif mask>0:
                    maskOnly.append(time)
           
        for i in (0,1):         
            hits[i].append(hitVal[i])
            misses[i].append(missVal[i])
            
    hitErr = [[np.std(val) for val in lst] for lst in hits]
    missErr = [[np.std(val) for val in lst] for lst in misses]      
            
    Rmed = list(map(np.median, hits[0]))  #one way
    Lmed = [np.median(x) for x in hits[1]]
    RmissMed = [np.median(x) for x in misses[0]]
    LmissMed = [np.median(x) for x in misses[1]]
    
    Rmean = [np.mean(x) for x in hits[0]]
    Lmean = [np.mean(x) for x in hits[1]]
    RmissMean = [np.mean(x) for x in misses[0]]
    LmissMean = [np.mean(x) for x in misses[1]]
    
    #max = np.max(np.mean(Rmed+Lmed))
    fig, ax = plt.subplots()
    ax.plot(np.unique(df[param]), Rmed, 'ro-', label='R hit',  alpha=.6, lw=3)
    ax.plot(np.unique(df[param]), Lmed, 'bo-', label='L hit', alpha=.6, lw=3)
   # ax.plot(np.unique(df[param]), RmissMed, 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
   # ax.plot(np.unique(df[param]), LmissMed, 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    plt.errorbar(np.unique(df[param]), Rmed, yerr=hitErr[0], c='r', alpha=.5)
    plt.errorbar(np.unique(df[param]), Lmed, yerr=hitErr[1], c='b', alpha=.5)
    ax.plot(10, np.median(maskOnly), marker='o', c='k')
    ax.set(title='Median Response Time From StimStart, by {}'.format(param), 
           xlabel=param.upper(), ylabel='Reaction Time (ms)')
    ax.set_xticks(np.unique(df[param]))
    a = ax.get_xticks().tolist()
    if param=='soa':
        a[0] = 'Mask Only'
        a[-1] = 'TargetOnly'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.plot(np.unique(df[param]), Rmean, 'ro-', label='R hit',  alpha=.6, lw=3)
    ax.plot(np.unique(df[param]), RmissMean, 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
    ax.plot(np.unique(df[param]), Lmean, 'bo-', label='L hit', alpha=.6, lw=3)
    ax.plot(np.unique(df[param]), LmissMean, 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    ax.plot(10, np.median(maskOnly), marker='o', c='k')
    ax.set(title='Mean Response Time From StimStart, by {}'.format(param), 
           xlabel=param.upper(), ylabel='Reaction Time (ms)')
    ax.set_xticks(np.unique(df[param]))
    a = ax.get_xticks().tolist()
    #a = [int(i) for i in a]     
    if param=='soa':
        a[0] = 'MaskOnly'
        a[-1] = 'Target Only'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()
    
    
    #err = [np.std(mean) for mean in Rmean]