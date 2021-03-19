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
from maskTaskAnalysis import MaskTaskData



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
        


exps = []
while True:
    f = fileIO.getFile('choose masking ephys experiment',fileType='*.hdf5')
    if f!='':
        obj = MaskTaskData()
        obj.loadFromHdf5(f)
        exps.append(obj)
    else:
        break
    

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


# plot response to optogenetic stimuluation during catch trials
psth = []
peakBaseRate = np.full(sum(obj.goodUnits.size for obj in exps),np.nan)
meanBaseRate = peakBaseRate.copy()
transientOptoResp = peakBaseRate.copy()
sustainedOptoResp = peakBaseRate.copy()
sustainedOptoRate = peakBaseRate.copy()
preTime = 0.5
postTime = 0.5
trialTime = [(obj.openLoopFrames+obj.responseWindowFrames)/obj.frameRate for obj in exps]
assert(all([t==trialTime[0] for t in trialTime]))
trialTime = trialTime[0]
windowDur = preTime+trialTime+postTime
binSize = 1/exps[0].frameRate
i = 0
for obj in exps:
    optoOnsetToPlot = np.nanmin(obj.optoOnset)
    for u in obj.goodUnits:
        spikeTimes = obj.units[u]['samples']/obj.sampleRate
        p = []
        for onset in [optoOnsetToPlot]: #np.unique(optoOnset[~np.isnan(optoOnset)]):
            trials = (obj.trialType=='catchOpto') & (obj.optoOnset==onset)
            startTimes = obj.frameSamples[obj.stimStart[trials]+int(onset)]/obj.sampleRate-preTime
            s,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=False)
            p.append(s)
        p = np.mean(np.concatenate(p),axis=0)
        psth.append(p)
        peakBaseRate[i] = p[t<preTime].max()
        meanBaseRate[i] = p[t<preTime].mean()
        transientOptoResp[i] = p[(t>=preTime) & (t<preTime+0.25)].max()-peakBaseRate[i]
        sustainedOptoRate[i] = p[(t>preTime+trialTime-0.35) & (t<preTime+trialTime-0.25)].mean()
        sustainedOptoResp[i] = sustainedOptoRate[i]-meanBaseRate[i]
        i += 1
t -= preTime
psth = np.array(psth)

fs = np.concatenate([obj.peakToTrough<=0.5 for obj in exps])

fig = plt.figure(figsize=(8,8))
excit = sustainedOptoResp>1
inhib = ((sustainedOptoResp<0) & (transientOptoResp<1))
transient = ~(excit | inhib) & (transientOptoResp>1)
noResp = ~(excit | inhib | transient)
gs = matplotlib.gridspec.GridSpec(4,2)
for i,j,clr,ind,lbl in zip((0,1,0,1,2,3),(0,0,1,1,1,1),'mgkkkk',(fs,~fs,excit,inhib,transient,noResp),('FS','RS','Excited','Inhibited','Transient','No Response')):
    ax = fig.add_subplot(gs[i,j])
    ax.plot(t,psth[ind].mean(axis=0),clr)
    ylim = plt.get(ax,'ylim')
    poly = np.array([(0,0),(trialTime-optoOnsetToPlot/obj.frameRate+0.1,0),(trialTime,ylim[1]),(0,ylim[1])])
    ax.add_patch(matplotlib.patches.Polygon(poly,fc='c',ec='none',alpha=0.25))
    n = str(np.sum(ind)) if j==0 else str(np.sum(ind & fs))+' FS, '+str(np.sum(ind & ~fs))+' RS'
    ax.text(1,1,lbl+' (n = '+n+')',transform=ax.transAxes,color=clr,ha='right',va='bottom')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    if (i==1 and j==0) or i==2:
        ax.set_xlabel('Time from LED onset (s)')
    ax.set_ylabel('Spikes/s')
plt.tight_layout()

fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(4,2)
for j,(xdata,xlbl) in enumerate(zip((obj.peakToTrough,obj.unitPos),('Spike peak to trough (ms)','Distance from tip (mm)'))):
    for i,(y,ylbl) in enumerate(zip((meanBaseRate,transientOptoResp,sustainedOptoResp,sustainedOptoRate),('Baseline rate','Transient opto response','Sustained opto response','Sustained opto rate'))):
        ax = fig.add_subplot(gs[i,j])
        for ind,clr in zip((fs,~fs),'mg'):
            ax.plot(xdata[ind],y[ind],'o',mec=clr,mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i in (1,2):
            ax.plot(plt.get(ax,'xlim'),[0,0],'--',color='0.6',zorder=0)
        if i==3:
            ax.set_xlabel(xlbl)
        if j==0:
            ax.set_ylabel(ylbl)
        if i==0 and j==0:
            for x,lbl,clr in zip((0.25,0.75),('FS','RS'),'mg'):
                ax.text(x,1.05,lbl,transform=ax.transAxes,color=clr,fontsize=12,ha='center',va='bottom')
plt.tight_layout()


# plot response to visual stimuli without opto
respThresh = 5 # stdev
stimLabels = ('targetOnly','maskOnly','mask')
respLabels = ('all','go','nogo')
cellTypeLabels = ('all','FS','RS')
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/obj.frameRate
ntrials = {stim: {side: {resp: {} for resp in respLabels} for side in ('left','right')} for stim in stimLabels}
psth = {cellType: {stim: {side: {resp: {} for resp in respLabels} for side in ('left','right')} for stim in stimLabels} for cellType in cellTypeLabels}
hasResp = copy.deepcopy(psth)
peakResp = copy.deepcopy(psth)
timeToPeak = copy.deepcopy(psth)
timeToFirstSpike = copy.deepcopy(psth)
for stim in stimLabels:
    for rd,side in zip((1,-1),('left','right')):
        for resp in respLabels:
            for obj in exps:
                fs = obj.peakToTrough<=0.5
                stimTrials = obj.trialType==stim if stim=='maskOnly' else (obj.trialType==stim) & (obj.rewardDir==rd)
                if resp=='all':
                    respTrials = np.ones(obj.ntrials,dtype=bool)
                else:
                    respTrials = ~np.isnan(obj.responseDir) if resp=='go' else np.isnan(obj.responseDir)
                for mo in np.unique(obj.maskOnset[stimTrials]):
                    moTrials = obj.maskOnset==mo
                    trials = stimTrials & respTrials & (obj.maskOnset==mo)
                    if mo not in ntrials[stim][side][resp]:
                        ntrials[stim][side][resp][mo] = 0
                    ntrials[stim][side][resp][mo] += trials.sum()
                    startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
                    for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
                        if mo not in psth[cellType][stim][side][resp]:
                            psth[cellType][stim][side][resp][mo] = []
                            hasResp[cellType][stim][side][resp][mo] = []
                            peakResp[cellType][stim][side][resp][mo] = []
                            timeToPeak[cellType][stim][side][resp][mo] = []
                            timeToFirstSpike[cellType][stim][side][resp][mo] = []
                        for u in obj.goodUnits[ct]:
                            spikeTimes = obj.units[u]['samples']/obj.sampleRate
                            p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                            t -= preTime
                            analysisWindow = (t>0.03) & (t<0.15)
                            p -= p[t<0].mean()
                            psth[cellType][stim][side][resp][mo].append(p)
                            hasResp[cellType][stim][side][resp][mo].append(p[analysisWindow].max() > respThresh*p[t<0].std())
                            peakResp[cellType][stim][side][resp][mo].append(p[analysisWindow].max())
                            timeToPeak[cellType][stim][side][resp][mo].append(t[np.argmax(p[analysisWindow])+np.where(analysisWindow)[0][0]])
                            lat = []
                            for st in startTimes:
                                firstSpike = np.where((spikeTimes > st+0.03) & (spikeTimes < st+0.15))[0]
                                if len(firstSpike)>0:
                                    lat.append(spikeTimes[firstSpike[0]]-st)
                                else:
                                    lat.append(np.nan)
                            timeToFirstSpike[cellType][stim][side][resp][mo].append(np.nanmedian(lat))

respCells = {cellType: np.array(hasResp[cellType]['targetOnly']['right']['all'][0]) | np.array(hasResp[cellType]['maskOnly']['right']['all'][0]) for cellType in cellTypeLabels}


xlim = [-0.1,0.4]
for ct,cellType in zip((np.ones(fs.size,dtype=bool),fs,~fs),cellTypeLabels):
    if cellType!='all':
        continue
    axs = []
    ymin = ymax = 0
    for resp in respLabels:
        fig = plt.figure(figsize=(10,5))
        fig.text(0.5,0.99,cellType+' (n='+str(respCells[cellType].sum())+' cells)',ha='center',va='top',fontsize=12)
        for i,(rd,side) in enumerate(zip((1,-1),('left','right'))):
            ax = fig.add_subplot(1,2,i+1)
            axs.append(ax)
            for stim,clr in zip(stimLabels,('k','0.5','r')):
                stimTrials = obj.trialType==stim if stim=='maskOnly' else (obj.trialType==stim) & (obj.rewardDir==rd)
                mskOn = np.unique(obj.maskOnset[stimTrials])
#                mskOn = [mskOn[-1]]
                if stim=='mask' and len(mskOn)>1:
                    cmap = np.ones((len(mskOn),3))
                    cint = 1/len(mskOn)
                    cmap[:,1:] = np.arange(0,1.01-cint,cint)[:,None]
                else:
                    cmap = [clr]
                for mo,c in zip(mskOn,cmap):
                    p = np.array(psth[cellType][stim][side][resp][mo])[respCells[cellType]]
                    m = np.mean(p,axis=0)
                    s = np.std(p,axis=0)/(len(p)**0.5)
                    lbl = 'target+mask, SOA '+str(round(1000*mo/obj.frameRate,1))+' ms' if stim=='mask' else stim
                    rlbl = '' if resp=='all' else ' '+resp
                    lbl += ' ('+str(ntrials[stim][side][resp][mo])+rlbl+' trials)'
#                    tme = t+2/obj.frameRate if stim=='maskOnly' else t
                    ax.plot(t,m,color=c,label=lbl)
        #            ax.fill_between(t,m+s,m-s,color=c,alpha=0.25)
                    ymin = min(ymin,np.min(m[(t>=xlim[0]) & (t<=xlim[1])]))
                    ymax = max(ymax,np.max(m[(t>=xlim[0]) & (t<=xlim[1])]))
            for s in ('right','top'):
                ax.spines[s].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from stimulus onset (s)')
            if i==0:
                ax.set_ylabel('Response (spikes/s)')
            ax.legend(loc='upper left',frameon=False,fontsize=8)
            ax.set_title('target '+side)
    for ax in axs:
        ax.set_ylim([1.05*ymin,1.05*ymax])
    

fig = plt.figure(figsize=(10,6))
gs = matplotlib.gridspec.GridSpec(2,2)
for j,(xdata,xlbl) in enumerate(zip((obj.peakToTrough,obj.unitPos),('Spike peak to trough (ms)','Distance from probe tip (mm)'))):
    for i,stim in enumerate(stimLabels[:2]):
        ax = fig.add_subplot(gs[i,j])
        for ct,cellType,clr in zip((fs,~fs),('FS','RS'),'mg'):
            for r,mfc in zip((~respCells[cellType],respCells[cellType]),('none',clr)):
                ax.plot(xdata[ct][r],peakResp[cellType][stim]['right']['all'][0][r],'o',mec=clr,mfc=mfc)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i==1:
            ax.set_xlabel(xlbl)
        if j==0:
            ax.set_ylabel('Response to '+stim[:stim.find('Only')]+' (spikes/s)')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,200],[0,200],'--',color='0.5')
amax = 0
for cellType,clr in zip(('FS','RS'),'mg'):
    for r,mfc in zip((~respCells[cellType],respCells[cellType]),('none',clr)):
        x,y = [peakResp[cellType][stim]['right']['all'][0][r] for stim in stimLabels[:2]]
        if any(x) and any(y):
            amax = max(amax,x.max(),y.max())
            ax.plot(x,y,'o',mec=clr,mfc=mfc,label=cellType)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.02*amax,1.02*amax])
ax.set_ylim([-0.02*amax,1.02*amax])
ax.set_aspect('equal')
ax.set_xlabel('Response to target (spikes/s)')
ax.set_ylabel('Response to mask (spikes/s)')
ax.legend()
plt.tight_layout()


fig = plt.figure(figsize=(8,6))
gs = matplotlib.gridspec.GridSpec(2,2)
for j,cellType in enumerate(('FS','RS')):
    for i,(ydata,ylbl) in enumerate(zip((timeToPeak,timeToFirstSpike),('Time to peak (ms)','Time to first spike (ms)'))):
        ax = fig.add_subplot(gs[i,j])
        for x,stim in enumerate(stimLabels):
            y = 1000*np.array(ydata[cellType][stim]['right']['all'][0])[respCells[cellType]]
            ax.plot(x+np.zeros(len(y)),y,'o',mec='0.5',mfc='none')
            m = np.nanmean(y)
            s = np.nanstd(y)/(np.sum(~np.isnan(y))**0.5)
            ax.plot(x,m,'ko')
            ax.plot([x,x],[m-s,m+s],'k')
            ax.text(x+0.1,m,str(round(m,1)),transform=ax.transData,ha='left',va='center')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(len(stimLabels)))
        if i==0:
            ax.set_xticklabels([])
            ax.set_title(cellType)
        else:
            ax.set_xticklabels(('target only','mask only','SOA 16.7 ms'))
        ax.set_xlim([-0.5,2.5])
        ax.set_ylim([30,150])
        if j==0:
            ax.set_ylabel(ylbl)
plt.tight_layout()


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



# plot response to visual stimuli with opto
preTime = 0.5
postTime = 0.5
windowDur = preTime+trialTime+postTime
binSize = 1/obj.frameRate
optOn = list(np.unique(obj.optoOnset[~np.isnan(obj.optoOnset)]))+[np.nan]
cmap = np.zeros((len(optOn),3))
cint = 1/(len(optOn)-1)
cmap[:-1,:2] = np.arange(0,1.01-cint,cint)[:,None]
cmap[:-1,2] = 1
fig = plt.figure(figsize=(6,10))
axs = []
naxs = 2+np.sum(np.unique(obj.maskOnset>0))
for stim in stimLabels:
    stimTrials = np.in1d(obj.trialType,(stim,stim+'Opto'))
    if stim!='maskOnly':
        stimTrials = stimTrials & (obj.rewardDir==-1)
    for mo in np.unique(obj.maskOnset[stimTrials]):
        ax = fig.add_subplot(naxs,1,len(axs)+1)
        axs.append(ax)
        moTrials = obj.maskOnset==mo
        for onset,clr in zip(optOn,cmap):
            trials = np.isnan(obj.optoOnset) if np.isnan(onset) else obj.optoOnset==onset
            trials = trials & stimTrials & moTrials
            startTimes = obj.frameSamples[obj.stimStart[trials]+obj.frameDisplayLag]/obj.sampleRate-preTime
            psth = []
            for u in obj.goodUnits[~fs & inhib]:
                spikeTimes = obj.units[u]['samples']/obj.sampleRate
                p,t = getPSTH(spikeTimes,startTimes,windowDur,binSize=binSize,avg=True)
                psth.append(p)
            t -= preTime
            m = np.mean(psth,axis=0)
            s = np.std(psth,axis=0)/(len(psth)**0.5)
            lbl = 'no opto' if np.isnan(onset) else str(int(round(1000*(onset-obj.frameDisplayLag)/obj.frameRate)))+' ms'
            ax.plot(t,m,color=clr,label=lbl)
#            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(-0.05,0.21,0.05))
        ax.set_xlim([-0.05,0.2])
        ax.set_ylabel('Spikes/s')
        title = 'SOA '+str(int(round(1000*mo/obj.frameRate)))+' ms' if stim=='mask' else stim
        ax.set_title(title)
        if len(axs)==1:
            ax.legend(loc='upper right',title='opto onset')
        elif len(axs)==naxs:
            ax.set_xlabel('Time from stimulus onset (s)')
ymin = min([plt.get(ax,'ylim')[0] for ax in axs]+[0])
ymax = max(plt.get(ax,'ylim')[1] for ax in axs)
for ax in axs:
    ax.set_ylim([ymin,ymax])
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

                
                

