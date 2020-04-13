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

def plot_by_param(df, selection='all', param1='soa', param2='trialLength', stat='Median', errorBars=False):    
    ''' 
        selection = 'all', 'hits', or 'misses'
        param1 = 'soa', 'targetContrast', or 'targetLength'
        param2 = 'trialLength', 'timeToMove', 'timeToOutcome'
        stat = 'Median' or 'Mean'
    '''

    matplotlib.rcParams['pdf.fonttype'] = 42
    sns.set_style('white')
    

    nonzeroRxns = df[(df['trialLength']!=df['trialLength'].max()) & 
                     (df['ignoreTrial']!=True) & (df['resp']!=0)]
    # ^ this excludes noResp trials and correct NoGos; soa=-1 are nogo trials 
    
    corrNonzero = nonzeroRxns[(nonzeroRxns['resp']==1) & (nonzeroRxns['nogo']==False)]
    missNonzero = nonzeroRxns[(nonzeroRxns['resp']==-1) & (nonzeroRxns['nogo']==False)]
    
    if len(corrNonzero)>0:
        v = corrNonzero.groupby(['rewDir', param1])[param2].describe()
        print('correct response times\n', v, '\n\n')
    if len(missNonzero)>0:
        y = missNonzero.groupby(['rewDir', param1])[param2].describe()
        print('incorrect response times\n', y)

 ### how to make this less bulky/redundant??     
    param_list = [x for x in np.unique(nonzeroRxns[param1]) if x >=0]   
 
    hits = [[],[]]  #R, L
    misses = [[],[]]
    maskOnly = []
  # change maskOnly filter to show turning direction
    
    for val in param_list:
        hitVal = [[],[]]
        missVal = [[],[]]
        for j, (time, p, resp, direc, mask) in enumerate(zip(
                nonzeroRxns[param2], nonzeroRxns[param1], nonzeroRxns['resp'], 
                nonzeroRxns['rewDir'], nonzeroRxns['maskContrast'])):
            if p==val:  
                if direc==1:       
                    if resp==1:
                        hitVal[0].append(time)  
                    else:
                        missVal[0].append(time)  
                elif direc==-1:   
                    if resp==1:
                        hitVal[1].append(time)  
                    else:
                        missVal[1].append(time)
                elif direc==0 and mask>0:
                    maskOnly.append(time)
           
        for i in (0,1):         
            hits[i].append(hitVal[i])
            misses[i].append(missVal[i])
            
    hitErr = [[np.std(val) for val in lst] for lst in hits]
    missErr = [[np.std(val) for val in lst] for lst in misses]      
            

    if stat=='Median':
        func=np.median
    else:
        func=np.mean
        
    avgHits = [[func(x) for x in side] for side in hits]   # 0=R, 1=L
    avgMisses = [[func(x) for x in side] for side in misses]
    
### plotting 
    
    if param2=='trialLength':
        label = 'Response Time' 
    elif param2 == 'timeToMove':
        label = 'Time to Initiate Movement'
    elif param2 == 'timeToOutcome':
        label = 'Reaction Time'

 
    fig, ax = plt.subplots()
    if selection=='all':
        ax.plot(param_list, avgHits[0], 'ro-', label= label + ' (R correct)',  alpha=.6, lw=3)
        ax.plot(param_list, avgHits[1], 'bo-', label='{} (L correct)'.format(label), alpha=.6, lw=3)
        ax.plot(param_list, avgMisses[0], 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
        ax.plot(param_list, avgMisses[1], 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    elif selection=='hits':
        ax.plot(param_list, avgHits[0], 'ro-', label='R hit',  alpha=.6, lw=3)
        ax.plot(param_list, avgHits[1], 'bo-', label='L hit', alpha=.6, lw=3)
    elif selection=='misses':   
        ax.plot(param_list, avgMisses[0], 'ro-', label='R miss', ls='--', alpha=.4, lw=2)
        ax.plot(param_list, avgMisses[1], 'bo-', label='L miss', ls='--', alpha=.4, lw=2)
   
    if errorBars:
        if selection=='hits'.lower():
            plt.errorbar(param_list, avgHits[0], yerr=hitErr[0], c='r', alpha=.5)
            plt.errorbar(param_list, avgHits[1], yerr=hitErr[1], c='b', alpha=.5)
        elif selection=='misses'.lower():
            plt.errorbar(param_list, avgMisses[0], yerr=missErr[0], c='r', alpha=.3)
            plt.errorbar(param_list, avgMisses[1], yerr=missErr[1], c='b', alpha=.3)
     
    if param1=='soa' and selection=='all':
        avgMaskOnly = func(maskOnly)    
        ax.plot(8, avgMaskOnly, marker='o', c='k')
        param_list[0] = 8
        
    
        
    ax.set(title='{} {} From StimStart, by {}'.format(stat,param2, param1), 
           xlabel=param1.upper() + ' (ms)', ylabel='Time from Stimulus Onset (ms)')
   # plt.suptitle((df.mouse + '   ' + df.date))  # need ot figure out loss of df metadata
    
    ax.set_xticks(param_list)   
    a = ax.get_xticks().tolist()
    if param1=='soa':
        a = [int(i) for i in a if i>=0]
        a[0] = 'Mask\nOnly'
        a[-1] = 'Target\nOnly'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()

 