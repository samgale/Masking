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

def plot_by_param(df, selection='all', param='soa', stat='Median', errorBars=False):    
    ''' 
        selection = 'all', 'hits', or 'misses'
        param = 'soa', 'targetContrast', or 'targetLength'
        stat = 'Median' or 'Mean'
    '''

    matplotlib.rcParams['pdf.fonttype'] = 42
    sns.set_style('white')

    nonzeroRxns = df[(df['trialLength']!=df['trialLength'].max()) & 
                     (df['ignoreTrial']!=True) & (df['resp']!=0)]
    # ^ this excludes noResp trials and correct NoGos; soa=-1 are nogo trials 
    
    corrNonzero = nonzeroRxns[(nonzeroRxns['resp']==1) & (nonzeroRxns['nogo']==False)]
    missNonzero = nonzeroRxns[(nonzeroRxns['resp']==-1) & (nonzeroRxns['nogo']==False)]
    
    param_list = [x for x in np.unique(nonzeroRxns[param]) if x >=0]
    
    
    miss = missNonzero.pivot_table(values='trialLength', index=param, columns='rewDir', 
                            margins=True, dropna=True)
    hit = corrNonzero.pivot_table(values='trialLength', index=param, columns='rewDir', 
                            margins=True, dropna=True)
    
    print('hits avg t \n', hit)
    print('\n' * 2)
    print('misses avg t \n', miss)


    # use the df to filter the trial by RewDir 
        # maybe use multiindex?? 
    
    y = corrNonzero.groupby(['rewDir', param])['trialLength'].describe()
    print(y)
    #y.to_excel("date_describe.xlsx")
    
    #to reduce bulk below; something like this?
    Rhit = corrNonzero[corrNonzero['rewDir']==1]
    avgs = Rhit.groupby('soa')['trialLength'].mean()
        

 ### how to make this less bulky/redundant??     
    
    hits = [[],[]]  #R, L
    misses = [[],[]]
    maskOnly = []
    
    for val in param_list:
        hitVal = [[],[]]
        missVal = [[],[]]
        for j, (time, p, resp, direc, mask) in enumerate(zip(
                nonzeroRxns['trialLength'], nonzeroRxns[param], nonzeroRxns['resp'], 
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
                elif mask>0:
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
    avgMaskOnly = func(maskOnly)

 
    fig, ax = plt.subplots()
    if selection=='all':
        ax.plot(param_list, avgHits[0], 'ro-', label='R hit',  alpha=.6, lw=3)
        ax.plot(param_list, avgHits[1], 'bo-', label='L hit', alpha=.6, lw=3)
        ax.plot(param_list, avgMisses[0], 'ro-', label='R miss', ls='--', alpha=.3, lw=2)
        ax.plot(param_list, avgMisses[1], 'bo-', label='L miss', ls='--', alpha=.3, lw=2)
    elif selection=='hits':
        ax.plot(param_list, avgHits[0], 'ro-', label='R hit',  alpha=.6, lw=3)
        ax.plot(param_list, avgHits[1], 'bo-', label='L hit', alpha=.6, lw=3)
    elif selection=='misses':   
        ax.plot(param_list, avgMisses[0], 'ro-', label='R miss', ls='--', alpha=.4, lw=2)
        ax.plot(param_list, avgMisses[1], 'bo-', label='L miss', ls='--', alpha=.4, lw=2)
   
    
    if errorBars==True:
        if selection=='hits'.lower():
            plt.errorbar(param_list, avgHits[0], yerr=hitErr[0], c='r', alpha=.5)
            plt.errorbar(param_list, avgHits[1], yerr=hitErr[1], c='b', alpha=.5)
        elif selection=='misses'.lower():
            plt.errorbar(param_list, avgMisses[0], yerr=missErr[0], c='r', alpha=.3)
            plt.errorbar(param_list, avgMisses[1], yerr=missErr[1], c='b', alpha=.3)
     
        
    ax.plot(8, avgMaskOnly, marker='o', c='k')
    
    ax.set(title='{} Response Time From StimStart, by {}'.format(stat, param), 
           xlabel=param.upper(), ylabel='Reaction Time (ms)')

    param_list[0] = 8
    ax.set_xticks(param_list)   #HOW to exclude nogos from plotting? 
    a = ax.get_xticks().tolist()
    if param=='soa':
        a = [int(i) for i in a if i>=0]
        a[0] = 'Mask\nOnly'
        a[-1] = 'Target\nOnly'
    ax.set_xticklabels(a)
    matplotlib.rcParams["legend.loc"] = 'best'
    ax.legend()

 