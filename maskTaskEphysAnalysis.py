# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:44 2019

@author: svc_ccg
"""

import copy
import pickle
import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
import fileIO



def getPSTH(spikes,startTimes,windowDur,binSize=0.01,avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts, bins[:-1]+binSize/2


def getSDF(spikes,startTimes,windowDur,sampInt=0.001,filt='exponential',filtWidth=0.005,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
        if filt in ('exp','exponential'):
            filtPts = int(5*filtWidth/sampInt)
            expFilt = np.zeros(filtPts*2)
            expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=filtWidth/sampInt,sym=False)
            expFilt /= expFilt.sum()
            sdf = scipy.ndimage.filters.convolve1d(counts,expFilt,axis=1)
        else:
            sdf = scipy.ndimage.filters.gaussian_filter1d(counts,filtWidth/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf,t[:-1]



obj = MaskTaskData()

obj.loadEphysData()

obj.loadBehavData()

obj.loadRFData()

obj.saveToHdf5()        


exps = []
for f in fileIO.getFiles('choose experiments',fileType='*.hdf5'):
    obj = MaskTaskData()
    obj.loadFromHdf5(f)
    exps.append(obj)


# plot response to visual stimuli without opto
trialTime = [(obj.openLoopFrames+obj.responseWindowFrames)/obj.frameRate for obj in exps]
assert(all([t==trialTime[0] for t in trialTime]))
trialTime = trialTime[0]

frameRate = 120
binSize = 1/exps[0].frameRate

fsThresh = 0.5
relThresh = 5 # stdev
absThresh = 1
stimLabels = ('targetOnly','maskOnly','mask','catch')
behavRespLabels = ('all','go','nogo')
cellTypeLabels = ('all','FS','RS')
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime

ntrials = {stim: {hemi: {resp: {} for resp in behavRespLabels} for hemi in ('ipsi','contra')} for stim in stimLabels}
psth = {cellType: {stim: {hemi: {resp: {} for resp in behavRespLabels} for hemi in ('ipsi','contra')} for stim in stimLabels} for cellType in cellTypeLabels}
hasResp = copy.deepcopy(psth)

for obj in exps:
    ephysHemi = ('contra','ipsi') if hasattr(obj,'hemi') and obj.hemi=='right' else ('ipsi','contra')
    fs = obj.peakToTrough<=fsThresh
    for stim in stimLabels:
        for rd,hemi in zip((1,-1),ephysHemi):
            stimTrials = obj.trialType==stim
            if stim in ('targetOnly','mask'):
                stimTrials = stimTrials & (obj.rewardDir==rd)
            for resp in behavRespLabels:
                if resp=='all':
                    respTrials = np.ones(obj.ntrials,dtype=bool)
                else:
                    respTrials = ~np.isnan(obj.responseDir) if resp=='go' else np.isnan(obj.responseDir)
                for mo in np.unique(obj.maskOnset[stimTrials]):
                    moTrials = obj.maskOnset==mo
                    trials = stimTrials & respTrials & (obj.maskOnset==mo)
                    if mo not in ntrials[stim][hemi][resp]:
                        ntrials[stim][hemi][resp][mo] = 0
                    ntrials[stim][hemi][resp][mo] += trials.sum()
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
                        if mo not in psth[cellType][stim][hemi][resp]:
                            psth[cellType][stim][hemi][resp][mo] = []
                            hasResp[cellType][stim][hemi][resp][mo] = []
                        for u in obj.goodUnits[ct]:
                            spikeTimes = obj.units[u]['samples']/obj.sampleRate
                            p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                            psth[cellType][stim][hemi][resp][mo].append(p)
                            t -= preTime
                            analysisWindow = (t>0.025) & (t<0.15)
                            b = p-p[t<0].mean()
                            hr = (b[analysisWindow].mean() > absThresh) & (b[analysisWindow].max() > relThresh*b[t<0].std())
                            hasResp[cellType][stim][hemi][resp][mo].append(hr)

respCells = {cellType: np.array(hasResp[cellType]['targetOnly']['contra']['all'][0]) | np.array(hasResp[cellType]['maskOnly']['contra']['all'][0]) for cellType in cellTypeLabels}
#respCells = {cellType: np.array(hasResp[cellType]['targetOnly']['contra']['all'][0]) for cellType in cellTypeLabels}

xlim = [-0.1,0.4]
for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
    if cellType!='all':
        continue
    axs = []
    ymin = ymax = 0
    for resp in ('all',): #behavRespLabels:
        fig = plt.figure(figsize=(10,5))
        fig.text(0.5,0.99,cellType+' (n='+str(respCells[cellType].sum())+' cells)',ha='center',va='top',fontsize=12)
        rewDir = (-1,1) if hasattr(obj,'hemi') and obj.hemi=='right' else (1,-1)
        for i,(rd,hemi) in enumerate(zip((1,-1),('ipsi','contra'))):
            ax = fig.add_subplot(1,2,i+1)
            axs.append(ax)
            for stim,clr in zip(stimLabels,('k','0.5','r','m')):
                stimTrials = obj.trialType==stim if stim=='maskOnly' else (obj.trialType==stim) & (obj.rewardDir==rd)
                mskOn = np.unique(obj.maskOnset[stimTrials])
                if stim=='mask' and len(mskOn)>1:
                    cmap = np.ones((len(mskOn),3))
                    cint = 1/len(mskOn)
                    cmap[:,1:] = np.arange(0,1.01-cint,cint)[:,None]
                else:
                    cmap = [clr]
                for mo,c in zip(mskOn,cmap):
                    p = np.array(psth[cellType][stim][hemi][resp][mo])[respCells[cellType]]
                    m = np.mean(p,axis=0)
                    s = np.std(p,axis=0)/(len(p)**0.5)
                    lbl = 'target+mask, SOA '+str(round(1000*mo/obj.frameRate,1))+' ms' if stim=='mask' else stim
                    rlbl = '' if resp=='all' else ' '+resp
                    lbl += ' ('+str(ntrials[stim][hemi][resp][mo])+rlbl+' trials)'
                    ax.plot(t,m,color=c,label=lbl)
#                    ax.fill_between(t,m+s,m-s,color=c,alpha=0.25)
                    ymin = min(ymin,np.min(m[(t>=xlim[0]) & (t<=xlim[1])]))
                    ymax = max(ymax,np.max(m[(t>=xlim[0]) & (t<=xlim[1])]))
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from stimulus onset (s)')
            if i==0:
                ax.set_ylabel('Response (spikes/s)')
            ax.legend(loc='upper left',frameon=False,fontsize=8)
            ax.set_title('target '+hemi)
    for ax in axs:
        ax.set_ylim([1.05*ymin,1.05*ymax])


# save psth
popPsth = {stim: {hemi: {} for hemi in ('ipsi','contra')} for stim in stimLabels}
for cellType in ('all',):
    for stim in stimLabels:
        for side,hemi in zip(('left','right'),('ipsi','contra')):
            p = psth[cellType][stim][side]['all']
            for mo in p.keys():
                popPsth[stim][hemi][mo] = np.mean(np.array(p[mo])[respCells[cellType]],axis=0)
popPsth['t'] = t
            
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(popPsth,open(pkl,'wb'))


# plot response to optogenetic stimuluation during catch trials
optoPsthExample = []
i = 0
for obj in exps:
    optoOnsetToPlot = np.nanmin(obj.optoOnset)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        for onset in [optoOnsetToPlot]:
            trials = (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
        t -= preTime
        optoPsthExample.append(p)
        i += 1
optoPsthExample = np.array(optoPsthExample)

optoPsth = []
baseRate = np.full(sum(obj.goodUnits.size for obj in exps),np.nan)
stdBaseRate = baseRate.copy()
transientOptoResp = baseRate.copy()
sustainedOptoResp = baseRate.copy()
sustainedOptoRate = baseRate.copy()
i = 0
for obj in exps:
    for u in obj.goodUnits:
        p = []
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        for onset in np.unique(obj.optoOnset[~np.isnan(obj.optoOnset)]):
            trials = (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            s,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
            p.append(s)
        t -= preTime
        p = np.mean(np.concatenate(p),axis=0)
        optoPsth.append(p)
        baseRate[i] = p[t<0].mean()
        stdBaseRate[i] = p[t<0].std()
        transientOptoResp[i] = p[(t>0) & (t<0.1)].max()-baseRate[i]
        sustainedOptoRate[i] = p[(t>0.1) & (t<0.5)].mean()
        sustainedOptoResp[i] = sustainedOptoRate[i]-baseRate[i]
        i += 1
optoPsth = np.array(optoPsth)

peakToTrough = np.concatenate([obj.peakToTrough for obj in exps])
unitPos = np.concatenate([obj.unitPos for obj in exps])
fs = peakToTrough < fsThresh

peakThresh = 3*stdBaseRate
sustainedThresh = stdBaseRate
excit = sustainedOptoResp>sustainedThresh
transient = ~excit & (transientOptoResp>peakThresh)
inhib = ~transient & (sustainedOptoResp<sustainedThresh)
noResp = ~(excit | inhib | transient)

for k,p in enumerate((optoPsthExample,optoPsth)):
    fig = plt.figure(figsize=(8,8))
    gs = matplotlib.gridspec.GridSpec(4,2)
    for i,j,units,clr,lbl in zip((0,1,0,1,2,3),(0,0,1,1,1,1),(fs,~fs,excit,inhib,transient,noResp),'mgkkkk',('FS','RS','Excited','Inhibited','Transient','No Response')):
        ax = fig.add_subplot(gs[i,j])
        ax.plot(t,p[units].mean(axis=0),clr)
        ylim = plt.get(ax,'ylim')
        if k==0:
            poly = np.array([(0,0),(trialTime-optoOnsetToPlot/obj.frameRate+0.1,0),(trialTime,ylim[1]),(0,ylim[1])])
            ax.add_patch(matplotlib.patches.Polygon(poly,fc='c',ec='none',alpha=0.25))
        n = str(np.sum(units)) if j==0 else str(np.sum(units & fs))+' FS, '+str(np.sum(units & ~fs))+' RS'
        ax.text(1,1,lbl+' (n = '+n+')',transform=ax.transAxes,color=clr,ha='right',va='bottom')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,trialTime+postTime])
        if (i==1 and j==0) or i==2:
            ax.set_xlabel('Time from LED onset (s)')
        ax.set_ylabel('Spikes/s')
    plt.tight_layout()

fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(4,2)
for j,(xdata,xlbl) in enumerate(zip((peakToTrough,unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
    for i,(y,ylbl) in enumerate(zip((baseRate,transientOptoResp,sustainedOptoResp,sustainedOptoRate),('Baseline rate','Transient opto response','Sustained opto response','Sustained opto rate'))):
        ax = fig.add_subplot(gs[i,j])
        xmin = xdata.min()
        xmax = xdata.max()
        xrng = xmax-xmin
        xlim = [xmin-0.05*xrng,xmax+0.05*xrng]
        if i in (1,2):
            ax.plot(xlim,[0,0],'--',color='0.6')
        for ind,clr in zip((fs,~fs),'mg'):
            ax.plot(xdata[ind],y[ind],'o',mec=clr,mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim(xlim)
        if i==3:
            ax.set_xlabel(xlbl)
        if j==0:
            ax.set_ylabel(ylbl)
        if i==0 and j==0:
            for x,lbl,clr in zip((0.25,0.75),('FS','RS'),'mg'):
                ax.text(x,1.05,lbl,transform=ax.transAxes,color=clr,fontsize=12,ha='center',va='bottom')
plt.tight_layout()


# plot response to visual stimuli with opto
optoOnset = list(np.unique(exps[0].optoOnset[~np.isnan(exps[0].optoOnset)]))+[np.nan]
optoOnsetPsth = {stim: {hemi: {onset: [] for onset in optoOnset} for hemi in ('ipsi','contra')} for stim in stimLabels}
for obj in exps:
    ephysHemi = ('contra','ipsi') if hasattr(obj,'hemi') and obj.hemi=='right' else ('ipsi','contra')
    for stim in stimLabels:
        for hemi,rd in zip(ephysHemi,(1,-1)):
            stimTrials = np.in1d(obj.trialType,(stim,stim+'Opto'))
            if stim in ('targetOnly','mask'):
                stimTrials = stimTrials & (obj.rewardDir==rd)
            for mo in np.unique(obj.maskOnset[stimTrials]):
                moTrials = obj.maskOnset==mo
                for onset in optoOnset:
                    trials = np.isnan(obj.optoOnset) if np.isnan(onset) else obj.optoOnset==onset
                    trials = trials & stimTrials & moTrials
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for u in obj.goodUnits:
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                        optoOnsetPsth[stim][hemi][onset].append(p)                      
t -= preTime
t *= 1000
                
units = ~(transient | excit) & respCells['all']
                
optoOnsetTicks = list(1000*(np.array(optoOnset[:-1])-exps[0].frameDisplayLag)/frameRate) + [100]
optoOnsetLabels = [str(int(round(onset))) for onset in optoOnsetTicks[:-1]] + ['no opto']

fig = plt.figure(figsize=(6,10))
gs = matplotlib.gridspec.GridSpec(len(stimLabels),2)
axs = []
cmap = np.zeros((len(optoOnset),3))
cint = 1/(len(optoOnset)-1)
cmap[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
cmap[:-1,2] = 1
for i,stim in enumerate(stimLabels):
    for j,hemi in enumerate(('ipsi','contra')):
        ax = fig.add_subplot(gs[i,j])
        axs.append(ax)
        for onset,clr,lbl in zip(optoOnset,cmap,optoOnsetLabels):
            p = np.array(optoOnsetPsth[stim][hemi][onset])[units]
            m = np.mean(p,axis=0)
            s = np.std(p,axis=0)/(len(psth)**0.5)
            lbl = lbl if np.isnan(onset) else lbl+' ms'
            ax.plot(t,m,color=clr,label=lbl)
#            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(-50,201,50))
        ax.set_xlim([-25,175])
        ax.set_ylabel('Spikes/s')
        title = 'target+mask (17 ms offset)' if stim=='mask' else stim
        ax.set_title(title)
        if i==0 and j==0:
            ax.legend(loc='upper left',title='opto onset',fontsize=8)
        elif i==len(stimLabels)-1:
            ax.set_xlabel('Time from stimulus onset (ms)')
ymin = min([plt.get(ax,'ylim')[0] for ax in axs]+[0])
ymax = max(plt.get(ax,'ylim')[1] for ax in axs)
for ax in axs:
    ax.set_ylim([ymin,ymax])
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
analysisWindow = (t>25) & (t<150)
for stim,clr in zip(stimLabels[:3],'ckb'):
    respMean = []
    respSem = []
    for onset in optoOnset:
        p = np.array(optoOnsetPsth[stim]['contra'][onset])[units]
        c = np.array(optoOnsetPsth['catch']['contra'][onset])[units]
        r = (p-c)[:,analysisWindow].sum(axis=1)*analysisWindow.sum()*binSize
        respMean.append(np.mean(r))
        respSem.append(np.std(r)/(len(r)**0.5))
    lbl = 'target + mask' if stim=='mask' else stim
    ax.plot(optoOnsetTicks,respMean,'o',color=clr,label=lbl)
    for x,m,s in zip(optoOnsetTicks,respMean,respSem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(optoOnsetTicks)
ax.set_xticklabels(optoOnsetLabels)
ax.set_xlim([8,108])
ax.set_xlabel('Opto onset relative to target onset (ms)')
ax.set_ylabel('Stimulus evoked spikes')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
for stim,clr in zip(('targetOnly','mask'),'cb'):
    mean = []
    sem = []
    for onset in optoOnset:
#        contra = np.array(optoOnsetPsth[stim]['contra'][onset])[units]
#        ipsi = np.array(optoOnsetPsth[stim]['ipsi'][onset])[units]
#        diff = (contra-ipsi)[:,analysisWindow].sum(axis=1)*analysisWindow.sum()*binSize
#        mean.append(np.mean(diff))
#        sem.append(np.std(diff)/(len(diff)**0.5))
        d = []
        ipsi,contra = [np.array(optoOnsetPsth[stim][hemi][onset])[units][:,analysisWindow].sum(axis=1)*analysisWindow.sum()*binSize for hemi in ('ipsi','contra')]
#        for i,c in zip(ipsi,contra):
#            if i==0 and c==0:
#                d.append(0.5)
#            else:
#                d.append(c/(c+i))
#        mean.append(np.mean(d))
#        sem.append(np.std(d)/(len(d)**0.5))
        mean.append(contra.mean()/(ipsi.mean()+contra.mean()))
    ax.plot(optoOnsetTicks[2:],mean[2:],'o',color=clr)
#    for x,m,s in zip(optoOnsetTicks[2:],mean[2:],sem[2:]):
#        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(optoOnsetTicks)
ax.set_xticklabels(optoOnsetLabels)
ax.set_xlim([8,108])
ax.set_xlabel('Opto onset relative to target onset (ms)')
ax.set_ylabel('Contra / (Contra + Ipsi)')
plt.tight_layout()


optoTrialPsth = {stim: {hemi: {onset: [] for onset in optoOnset} for hemi in ('ipsi','contra')} for stim in stimLabels}
i = 0
for obj in exps:
    ephysHemi = ('contra','ipsi') if hasattr(obj,'hemi') and obj.hemi=='right' else ('ipsi','contra')
    for stim in stimLabels:
        for hemi,rd in zip(ephysHemi,(1,-1)):
            stimTrials = np.in1d(obj.trialType,(stim,stim+'Opto'))
            if stim in ('targetOnly','mask'):
                stimTrials = stimTrials & (obj.rewardDir==rd)
            for mo in np.unique(obj.maskOnset[stimTrials]):
                moTrials = obj.maskOnset==mo
                for onset in optoOnset:
                    trials = np.isnan(obj.optoOnset) if np.isnan(onset) else obj.optoOnset==onset
                    trials = trials & stimTrials & moTrials
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    p = []
                    for u in obj.goodUnits:
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        s,_ = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
                        p.append(s)
                    optoTrialPsth[stim][hemi][onset].append(np.array(p)[units[i:i+len(obj.goodUnits)]].mean(axis=0))
    i += len(obj.goodUnits)

analysisWindow = (t>25) & (t<100)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([8,108],[0.5,0.5],'k--')
for stim,clr in zip(('targetOnly','mask'),'cb'):
    mean = []
    sem = []
    for onset in optoOnset:
        d = []
        for ind in range(len(exps)):
            ipsi,contra = [optoTrialPsth[stim][hemi][onset][ind][:,analysisWindow].sum(axis=1)*analysisWindow.sum()*binSize for hemi in ('ipsi','contra')]
            for c in contra:
                b = []
                for i in ipsi:
                    if i==0 and c==0:
                        pass
                        b.append(0.5)
                    else:
                        b.append(c/(c+i))
                d.append(np.mean(b))
        mean.append(np.mean(d))
        sem.append(np.std(d)/(len(d)**0.5))
    if stim=='targetOnly':
        firstValid = 3
    elif stim=='mask':
        firstValid = 2
    else:
        firstValid = 0
    ax.plot(optoOnsetTicks[firstValid:-1],mean[firstValid:-1],color=clr)
    ax.plot(optoOnsetTicks[-1],mean[-1],'o',color=clr,label=lbl)
    for x,m,s in zip(optoOnsetTicks[firstValid:],mean[firstValid:],sem[firstValid:]):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(optoOnsetTicks)
ax.set_xticklabels(optoOnsetLabels)
ax.set_xlim([8,108])
ax.set_xlabel('Opto onset relative to target onset (ms)')
ax.set_ylabel('Fraction of spikes contralateral to target')
plt.tight_layout()



# rf mapping
azi,ele = [np.unique(p) for p in obj.rfStimPos.T]
ori = obj.rfOris
contrast = np.unique(obj.rfStimContrast)
dur = np.unique(obj.rfStimFrames)
binSize = 1/obj.frameRate
preTime = 0.1
postTime = 0.6
nbins = np.arange(0,preTime+postTime+binSize,binSize).size-1
rfMap = np.zeros((obj.goodUnits.size,nbins,ele.size,azi.size,ori.shape[0],contrast.size,dur.size))
for i,y in enumerate(ele):
    eleTrials = obj.rfStimPos[:,1]==y
    for j,x in enumerate(azi):
        aziTrials = obj.rfStimPos[:,0]==x
        for k,o in enumerate(ori):
            oriTrials = obj.rfStimOri[:,0]==o[0]
            if np.isnan(o[1]):
                oriTrials = oriTrials & np.isnan(obj.rfStimOri[:,1])
            else:
                oriTrials = oriTrials & (obj.rfStimOri[:,1]==o[1])
            for l,c in enumerate(contrast):
                contrastTrials = obj.rfStimContrast==c
                for m,d in enumerate(dur):
                    trials = eleTrials & aziTrials & oriTrials & contrastTrials & (obj.rfStimFrames==d)
                    startTimes = obj.frameSamples[obj.rfStimStart[trials]]/obj.sampleRate
                    for n,u in enumerate(obj.goodUnits):
                        spikeTimes = obj.units[u]['samples']/obj.sampleRate
                        p,t = getPSTH(spikeTimes,startTimes-preTime,preTime+postTime,binSize=binSize,avg=True)
                        t -= preTime
#                        p -= p[t<0].mean()
                        rfMap[n,:,i,j,k,l,m] = p


fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(ele.size,azi.size)
for i,y in enumerate(ele):
    for j,x in enumerate(azi):
        ax = fig.add_subplot(gs[ele.size-1-i,j])
        
        ax.plot(t,rfMap.mean(axis=(0,4,5,6))[:,i,j],'k')
        
#        for k,d in enumerate(dur):
#            ax.plot(t,rfMap.mean(axis=(0,4,5))[:,i,j,k],'k')
            
#        for k,c in enumerate(contrast):
#            ax.plot(t,rfMap.mean(axis=(0,4,6))[:,i,j,k],'k')
        
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==2].mean(axis=(0,4,5,6))[:,i,j],'k',label='17 ms stim')
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==6].mean(axis=(0,4,5,6))[:,i,j],'r',label='50 ms stim')
            
        # target
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==2].mean(axis=(0,5,6))[:,i,j,0],'k')
        
        # mask
#        ax.plot(t,rfMap[:,:,:,:,:,contrast==0.4][:,:,:,:,:,:,dur==6].mean(axis=(0,5,6))[:,i,j,4],'k')
        
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        if i==0 and j==1:
            ax.set_xlabel('Time from stim onset (s)')
        else:
            ax.set_xticklabels([])
        if i==1 and j==0:
            ax.set_ylabel('Spikes/s')
        else:
            ax.set_yticklabels([])
        if i==ele.size-1 and j==azi.size-1:
            ax.legend(loc='upper right')
        ax.set_xticks([0,0.5])
        ax.set_yticks([0,10,20,30])
        ax.set_ylim([0,35])
plt.tight_layout()


plt.figure()
for i,o in enumerate(ori):
    plt.plot(t,rfMap.mean(axis=(0,2,3,5,6))[:,i],label=o)
plt.legend()


# spike rate autocorrelation
corrWidth = 50
corr = np.zeros((sum([len(obj.goodUnits) for obj in exps]),corrWidth*2+1))
uind = 0
for obj in exps:
    starts = np.round(1000*np.concatenate(([0],obj.frameSamples[obj.stimStart]/obj.sampleRate+obj.responseWindowFrames/obj.frameRate))).astype(int)
    stops = np.round(np.concatenate((obj.frameSamples[obj.stimStart]/obj.sampleRate,[obj.frameSamples[-1]/obj.sampleRate]))*1000-corrWidth).astype(int)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate*1000
        c = np.zeros(corrWidth*2+1)
        n = 0
        for i,j in zip(starts,stops):
            spikes = spikeTimes[(spikeTimes>i) & (spikeTimes<j)]
            for s in spikes:
                bins = np.arange(s-corrWidth-0.5,s+corrWidth+1)
                c += np.histogram(spikes,bins)[0]
                n += 1
        c /= n
        corr[uind] = c
        uind += 1
        print(str(uind)+'/'+str(corr.shape[0]))
        

expfunc = lambda x,a,tau,c: a*np.exp(-x/tau)+c

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
t = np.arange(corrWidth*2+1)-corrWidth
x0 = 3
fitInd = slice(corrWidth+x0,None)
fitx = t[fitInd]-x0
#tau = []
#for c in corr:
#    if not np.all(np.isnan(c)):
#        ax.plot(t,c,color='0.5')
#        fitParams = scipy.optimize.curve_fit(expfunc,fitx,c[fitInd])[0]
#        tau.append(fitParams[1])
#        ax.plot(t[fitInd],expfunc(fitx,*fitParams),color=[1,0.5,0.5])
m = np.nanmean(corr,axis=0)
ax.plot(t,m,'k')
fitParams = scipy.optimize.curve_fit(expfunc,fitx,m[fitInd])[0]
tauMean = fitParams[1]
#ax.plot(t[fitInd],expfunc(fitx,*fitParams),'r')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
#ax.set_ylim([0,1.05*np.nanmax(corr[:,t!=0])])
ax.set_xlim([0,corrWidth])
ax.set_ylim([0,0.04])
ax.set_yticks([0,0.01,0.02,0.03,0.04])
ax.set_xlabel('Lag (ms)')
ax.set_ylabel('Autocorrelation')
plt.tight_layout()

                
                

