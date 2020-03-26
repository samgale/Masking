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
    
    miss = missNonzero.pivot_table(values='trialLength', index='{}'.format(param), columns='rewDir', 
                            margins=True, dropna=True)
    hit = corrNonzero.pivot_table(values='trialLength', index='{}'.format(param), columns='rewDir', 
                            margins=True, dropna=True)
    
    #hit.plot(title='hits')
    #miss.plot(title='misses')
    print('hits \n', hit)
    print('\n' * 2)
    print('misses \n', miss)


 ### how to make this less bulky/redundant??     
    
    hits = [[],[]]  #R, L
    misses = [[],[]]
    maskOnly = []
    
    for val in np.unique(df['{}'.format(param)]):
        hitVal = [[],[]]
        missVal = [[],[]]
        for j, (time, p, resp, direc) in enumerate(zip(
                nonzeroRxns['trialLength'], nonzeroRxns['{}'.format(param)], nonzeroRxns['resp'], 
                nonzeroRxns['rewDir'])):
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
                else:
                    maskOnly.append(time)
           
        for i in (0,1):         
            hits[i].append(hitVal[i])
            misses[i].append(missVal[i])
            
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
    ax.plot(np.unique(df['{}'.format(param)]), Rmed, 'ro-', label='R hit',  alpha=.6, lw=3)
    ax.plot(np.unique(df['{}'.format(param)]), RmissMed, 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
    ax.plot(np.unique(df['{}'.format(param)]), Lmed, 'bo-', label='L hit', alpha=.6, lw=3)
    ax.plot(np.unique(df['{}'.format(param)]), LmissMed, 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    #ax.plot(0, np.median(maskOnly), marker='o', c='k')
    ax.set(title='Median Response Time From StimStart, by {}'.format(param), 
           xlabel='{}'.format(param), ylabel='Reaction Time (ms)')
    ax.set_xticks(np.unique(df['{}'.format(param)]))
    a = ax.get_xticks().tolist()
    #a = [int(i) for i in a]     
    if param=='soa':
        a[0] = ''
        a[-1] = 'TargetOnly'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.plot(np.unique(df['{}'.format(param)]), Rmean, 'ro-', label='R hit',  alpha=.6, lw=3)
    ax.plot(np.unique(df['{}'.format(param)]), RmissMean, 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
    ax.plot(np.unique(df['{}'.format(param)]), Lmean, 'bo-', label='L hit', alpha=.6, lw=3)
    ax.plot(np.unique(df['{}'.format(param)]), LmissMean, 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    #ax.plot(0, np.median(maskOnly), marker='o', c='k')
    ax.set(title='Mean Response Time From StimStart, by {}'.format(param), 
           xlabel='{}'.format(param), ylabel='Reaction Time (ms)')
    ax.set_xticks(np.unique(df['{}'.format(param)]))
    a = ax.get_xticks().tolist()
    #a = [int(i) for i in a]     
    if param=='soa':
        a[0] = 'MaskOnly'
        a[-1] = 'Target Only'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()
    
    
    #err = [np.std(mean) for mean in Rmean]