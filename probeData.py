# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

from __future__ import division
import datetime, h5py, json, math, ntpath, os, shelve, shutil
import numpy as np
import scipy.ndimage.filters
import scipy.optimize
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt 
from matplotlib import gridspec
from matplotlib import cm
from PyQt4 import QtGui
from astropy.convolution import Gaussian2DKernel, convolve
import pandas


dataDir = r'C:\Users\SVC_CCG\Desktop\Data'

class probeData():
    
    def __init__(self):
        self.recording = 0
        self.TTLChannelLabels = ['VisstimOn', 'CamExposing', 'CamSaving', 'OrangeLaserShutter']
        self.channelMapFile = r'C:\Users\SVC_CCG\Documents\PythonScripts\imec_channel_map.prb'
        self.wheelChannel = 134
        self.diodeChannel = 135
        self.sampleRate = 30000     


    def loadKwd(self, filePath):
                
        f = h5py.File(filePath, 'r')
        datDict = {}
        datDict['info'] = f['recordings'][str(self.recording)].attrs
        
        datDict['data'] = f['recordings'][str(self.recording)]['data']
        datDict['gains'] = f['recordings'][str(self.recording)]['application_data']['channel_bit_volts'][:]
        
        datDict['sampleRate'] = datDict['info']['sample_rate']
        datDict['startTime'] = datDict['info']['start_time']
        
        datDict['firstAnalogSample'] = f['recordings'][str(self.recording)]['application_data']['timestamps'][0][0]
        
        return datDict
        
        
    def loadExperiment(self, loadRunningData=False):
        self.kwdFileList, nsamps = getKwdInfo()
        filelist = self.kwdFileList
        filePaths = [os.path.dirname(f) for f in filelist]        
        
        self.d = []
        for index, f in enumerate(filelist):
            ff = os.path.basename(os.path.dirname(f))
            ff = ff.split('_')[-1]  
            datDict = self.loadKwd(f)
            datDict['protocolName'] = ff
            datDict['numSamples'] = nsamps[index]
            self.d.append(datDict)
                    
            
        self.getSingleUnits(fileDir=os.path.dirname(filePaths[0]))
        self.mapChannels()
        self.visstimData = {}
        self.behaviorData = {}
        self.TTL = {}
        for pro, proPath in enumerate(filePaths):
            files = os.listdir(proPath)
            
            visStimFound = False
            eyeDataFound = False
            self.behaviorData[str(pro)] = {}
            for f in files:
                if 'VisStim' in f:
                    self.getVisStimData(os.path.join(proPath, f), protocol=pro)
                    visStimFound = True
                    continue
            
                #load eye tracking data
                if 'MouseEyeTracker' in f:  
                    self.getEyeTrackData(os.path.join(proPath, f), protocol=pro)
                    eyeDataFound = True
                    continue
                
            ttlFile = [f for f in files if f.endswith('kwe')][0]             
            self.getTTLData(filePath=os.path.join(proPath, ttlFile), protocol=pro)
            
            if loadRunningData:
                self.behaviorData[str(pro)]['running'] = self.decodeWheel(self.d[pro]['data'][::500, self.wheelChannel]*self.d[pro]['gains'][self.wheelChannel])
            
            if not visStimFound:
                print('No vis stim data found for ' + os.path.basename(proPath))
            if not eyeDataFound:
                print('No eye tracking data found for ' + os.path.basename(proPath))
            
        
    def getTTLData(self, filePath=None, protocol=0):
        
        if filePath is None:
            ttlFileDir = self.filePath[:self.filePath.rfind('/')]
            filelist = os.listdir(ttlFileDir)
            filePath = ttlFileDir + '/' + [f for f in filelist if f.endswith('kwe')][0]
        
        
        f = h5py.File(filePath, 'r')
        recordingID = f['event_types']['TTL']['events']['recording'][:]
        eventChannels = f['event_types']['TTL']['events']['user_data']['event_channels'][recordingID==self.recording]        
        edges = f['event_types']['TTL']['events']['user_data']['eventID'][recordingID==self.recording]
        timeSamples = f['event_types']['TTL']['events']['time_samples'][recordingID==self.recording]
        
        self.TTLChannels = np.unique(eventChannels)        
        self.TTL[str(protocol)] = {}
        for chan in self.TTLChannels:
            eventsForChan = np.where(eventChannels == chan)
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]] = {}
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['rising'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges == 1))] - self.d[protocol]['firstAnalogSample']
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['falling'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges ==0))] - self.d[protocol]['firstAnalogSample']
        
        if str(protocol) in self.visstimData:
            if not hasattr(self, 'frameSamples'):
                self.alignFramesToDiode(protocol=protocol)
    
    
    def getVisStimData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = getFile()
        
        dataFile = h5py.File(filePath)        
        self.visstimData[str(protocol)] = {}
        for params in dataFile:
            if dataFile[params].size > 1:
                self.visstimData[str(protocol)][params] = dataFile[params][:]
            else:
                self.visstimData[str(protocol)][params] = dataFile[params][()]   
    
    
    def getEyeTrackData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = getFile()
        
        dataFile = h5py.File(filePath)
        frameTimes = dataFile['frameTimes'][:]
        frameInt = 1/60.0
        nRealFrames = round(frameTimes[-1]/frameInt)+1
        frameInd = np.round(frameTimes/frameInt).astype(int)
        
        eyeDict = {}
        firstFrameIndex = np.where(self.TTL[str(protocol)]['CamExposing']['rising'][:] < self.TTL[str(protocol)]['CamSaving']['rising'][0])[0][-1]
        eyeTime = self.TTL[str(protocol)]['CamExposing']['rising'][firstFrameIndex:firstFrameIndex + nRealFrames] - self.d[protocol]['startTime']
        eyeDict['samples'] = eyeTime[eyeTime<self.d[protocol]['numSamples']]         
                
        for param in ('pupilArea','pupilX','pupilY'):
            eyeDict[param] = np.full(nRealFrames,np.nan)       
            eyeDict[param][frameInd] = dataFile[param][:]
            eyeDict[param] = eyeDict[param][0:eyeDict['samples'].size]
        
        self.behaviorData[str(protocol)]['eyeTracking'] = eyeDict
    
    
    def alignFramesToDiode(self, frameSampleAdjustment = None, plot = False, protocol=0):
        if frameSampleAdjustment is None:
            frameSampleAdjustment = np.round((4.5/60.0) * 30000) 
        self.visstimData[str(protocol)]['frameSamples'] =  (self.TTL[str(protocol)]['VisstimOn']['falling'] + frameSampleAdjustment).astype(int)
        self.visstimData[str(protocol)]['frameSamples'] = (self.TTL[str(protocol)]['VisstimOn']['falling'] + frameSampleAdjustment).astype(int)
        self._frameSampleAdjustment = frameSampleAdjustment
        
        if plot:
            plt.figure()
            plt.plot(self.data[str(protocol)]['data'][:self.visstimData[str(protocol)]['frameSamples'][10], self.diodeChannel])
            plt.plot(self.visstimData[str(protocol)]['frameSamples'][:10], np.ones(10) * np.max(self.data[str(protocol)]['data'][:self.visstimData[str(protocol)]['frameSamples'][10], self.diodeChannel]), 'go')
            
            plt.figure()
            plt.plot(self.data[str(protocol)]['data'][self.visstimData[str(protocol)]['frameSamples'][-10]:, self.diodeChannel])
            plt.plot(self.visstimData[str(protocol)]['frameSamples'][-10:] - self.visstimData[str(protocol)]['frameSamples'][-10], np.ones(10) * np.max(self.data[str(protocol)]['data'][self.visstimData[str(protocol)]['frameSamples'][-10]:, self.diodeChannel]), 'go')
    
    
    def mapChannels(self):
        f = open(self.channelMapFile, 'r') 
        fdict = json.load(f)
        self.channelMapping = np.array(fdict['0']['mapping'])
        self.channelMapping = self.channelMapping[np.where(self.channelMapping > 0)] - 1
    
    
    def decodeWheel(self, wheelData, kernelLength = 0.5, wheelSampleRate = 60.0):
    
        sampleRate = wheelSampleRate
        wheelData = wheelData - np.min(wheelData)
        wheelData = 2*np.pi*wheelData/np.max(wheelData)
          
        smoothFactor = sampleRate/60.0       
        angularWheelData = np.arctan2(np.sin(wheelData), np.cos(wheelData))
        angularWheelData = np.convolve(angularWheelData, np.ones(smoothFactor), 'same')/smoothFactor
        
        artifactThreshold = (100.0/sampleRate)/7.6      #reasonable bound for how far (in radians) a mouse could move in one sample point (assumes top speed of 100 cm/s)
        angularDisplacement = (np.diff(angularWheelData) + np.pi)%(2*np.pi) - np.pi
        angularDisplacement[np.abs(angularDisplacement) > artifactThreshold ] = 0
        wheelData = np.convolve(angularDisplacement, np.ones(kernelLength*sampleRate), 'same')/(kernelLength*sampleRate)
        wheelData *= 7.6*sampleRate
        wheelData = np.insert(wheelData, 0, wheelData[0])
        
        return wheelData
        
    def filterChannel(self, chan, cutoffFreqs, protocol=0):
        Wn = np.array(cutoffFreqs)/(self.sampleRate/2)        
        b,a = scipy.signal.butter(4, Wn, btype='bandpass')
        return scipy.signal.filtfilt(b, a, self.data[str(protocol)]['data'][:, chan])
        
    
    def thresholdChannel(self, chan, threshold, direction = -1, refractory = None, filterFreqs = None, protocol=0):
        
        if filterFreqs is not None:
            data = direction * self.filterChannel(chan, filterFreqs)
        else:
            data = direction * self.data[str(protocol)]['data'][:, chan]
        
        threshold = direction * threshold
        spikeTimes = np.array(np.where(data > threshold)[0])
        
        if refractory is None:
            refractory = 1.0/self.sampleRate
        
        if spikeTimes.size > 0:
            ISI = np.diff(spikeTimes)
            goodISI = np.array(np.where(ISI > refractory*self.sampleRate)[0]) + 1
            goodISI = np.insert(goodISI, 0, 0)
            spikeTimes = spikeTimes[goodISI]
        return spikeTimes
        
        
    def computeFiringRate(self, spikeTimes, kernelLength = 0.05, protocol=0):
        fr = np.zeros(self.d[protocol]['numSamples'])
        fr[spikeTimes] = 1
        fr = np.convolve(fr, np.ones(kernelLength*self.sampleRate), 'same')/(kernelLength)
        return fr    
        
   
    def triggeredAverage(self, dataToAlign, alignmentPoints, win = [0.1, 0.1], sampleRate = None):
        if sampleRate is None:
            sampleRate = self.sampleRate
        aligned = np.full([sampleRate*(win[0] + win[1]), len(alignmentPoints)], np.nan)
        for index, point in enumerate(alignmentPoints):
            try:            
                aligned[:, index] = dataToAlign[point - win[0]*sampleRate : point + win[1]*sampleRate]
            except:
                continue
        return aligned
     
               
    def findSpikesPerTrial(self, trialStarts, trialEnds, spikes): 
        spikesPerTrial = np.zeros(trialStarts.size)
        for trialNum in range(trialStarts.size):
            spikesPerTrial[trialNum] = np.count_nonzero(np.logical_and(spikes>=trialStarts[trialNum],spikes<=trialEnds[trialNum]))
        return spikesPerTrial
        
            
    def findRF(self, units=None, usePeakResp = True, sigma = 1, plot = True, minLatency = 0.05, maxLatency = 0.15, trials = None, protocol=None, fit=True, useCache=False, saveTag=''):

        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('sparseNoise')
        protocol = str(protocol)
        
        if plot:        
            fig = plt.figure(figsize = (10, 3*len(units)), facecolor='w')
            gs = gridspec.GridSpec(len(units), 6)
            
        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
        
        minLatencySamples = minLatency*self.sampleRate
        maxLatencySamples = maxLatency*self.sampleRate
   
        xpos = np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][trials, 0])
        ypos = np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][trials, 1])
        
        posHistory = self.visstimData[str(protocol)]['boxPositionHistory'][trials]
        colorHistory = self.visstimData[str(protocol)]['boxColorHistory'][trials, 0]        
        gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
        
        onVsOff = np.full(len(units),np.nan)
        respNormArea = np.full((len(units),2),np.nan)
        respHalfWidth = np.copy(respNormArea)
        if fit:
            onCenters = np.full((len(units),2),np.nan)
            offCenters = np.copy(onCenters)
            
        sdfSampInt = 0.001
        sdfSigma = 0.01
        sdfSamples = minLatencySamples+2*maxLatencySamples
        
        for uindex, unit in enumerate(units):
            if ('rf' + saveTag) in self.units[unit] and useCache:
                gridOnSpikes = self.units[unit]['rf' + saveTag]['on']
                gridOffSpikes = self.units[unit]['rf' + saveTag]['off']
                gridOnSpikes_filter = self.units[unit]['rf' + saveTag]['on_filter']
                gridOffSpikes_filter = self.units[unit]['rf' + saveTag]['off_filter']
                xpos = self.units[unit]['rf' + saveTag]['xpos']
                ypos = self.units[unit]['rf' + saveTag]['ypos']
                onFit = self.units[unit]['rf' + saveTag]['onFit']
                offFit = self.units[unit]['rf' + saveTag]['offFit']
            else:
                self.units[unit]['rf' + saveTag] = {}
                spikes = self.units[unit]['times'][str(protocol)]
                gridOnSpikes = np.zeros((ypos.size,xpos.size))
                gridOffSpikes = np.zeros_like(gridOnSpikes)
                sdfOn = np.zeros((ypos.size,xpos.size,round(sdfSamples/self.sampleRate/sdfSampInt)))
                sdfOff = np.zeros_like(sdfOn)
                for i,y in enumerate(ypos):
                    for j,x in enumerate(xpos):
                        po = np.logical_and(posHistory[:, 1] == y,posHistory[:, 0] == x)
                        posOnTrials = np.logical_and(po, colorHistory == 1)
                        posOffTrials = np.logical_and(po, colorHistory == -1)
                        
                        posOnFrames = self.visstimData[protocol]['stimStartFrames'][trials][posOnTrials]
                        posOffFrames = self.visstimData[protocol]['stimStartFrames'][trials][posOffTrials]
                        
                        posOnSamples = self.visstimData[protocol]['frameSamples'][posOnFrames]
                        posOffSamples = self.visstimData[protocol]['frameSamples'][posOffFrames]
                        
                        for p in posOnSamples:
                            gridOnSpikes[i,j] += np.count_nonzero(np.logical_and(spikes>=p+minLatencySamples,spikes<p+maxLatencySamples))
                        
                        for p in posOffSamples:
                            gridOffSpikes[i,j] += np.count_nonzero(np.logical_and(spikes>=p+minLatencySamples,spikes<p+maxLatencySamples))
                        
                        gridOnSpikes[i,j] /= posOnSamples.size
                        gridOffSpikes[i,j] /= posOffSamples.size
                        
                        sdfOn[i,j,:],sdfTime = self.getMeanSDF(spikes,posOnSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                        sdfOff[i,j,:],_ = self.getMeanSDF(spikes,posOffSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                    
                # convert spike count to spike rate
                gridOnSpikes /= maxLatency-minLatency
                gridOffSpikes /= maxLatency-minLatency
                
                inAnalysisWindow = np.logical_and(sdfTime>minLatency*2,sdfTime<minLatency+maxLatency)
                if usePeakResp:
                    gridOnSpikes = np.max(sdfOn[:,:,inAnalysisWindow],axis=2)
                    gridOffSpikes = np.max(sdfOff[:,:,inAnalysisWindow],axis=2)
                
                gaussianKernel = Gaussian2DKernel(stddev=sigma)
                gridOnSpikes_filter = convolve(gridOnSpikes, gaussianKernel, boundary='extend')
                gridOffSpikes_filter = convolve(gridOffSpikes, gaussianKernel, boundary='extend')
#                gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOnSpikes, sigma)
#                gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOffSpikes, sigma)
                
                pixPerDeg = self.visstimData[str(protocol)]['pixelsPerDeg']
                elev, azim = ypos/pixPerDeg, xpos/pixPerDeg
                onFit = offFit = None
                if fit:
                    fitParams = []
                    for d in (gridOnSpikes_filter,gridOffSpikes_filter):
                        # params: x0 , y0, sigX, sigY, theta, amplitude
                        data = np.copy(d)-d.min()
                        i,j = np.unravel_index(np.argmax(data),data.shape)
                        sigmaGuess = (azim[1]-azim[0])*(np.count_nonzero(data>0.5*data.max())**0.5)
                        initialParams = (azim[j],elev[i],sigmaGuess,sigmaGuess,0,data.max())
                        if not np.any(np.isnan(data)):
                            fitParams.append(fitGauss2D(azim,elev,data,initialParams))
                    onFit,offFit = fitParams
                    if onFit is not None and gridExtent[0]<onFit[0]<gridExtent[2] and gridExtent[1]<onFit[1]<gridExtent[3]:
                        onCenters[uindex,:] = onFit[0:2]
                    if offFit is not None and gridExtent[0]<offFit[0]<gridExtent[2] and gridExtent[1]<offFit[1]<gridExtent[3]:
                        offCenters[uindex,:] = offFit[0:2]
                
                onMax = gridOnSpikes_filter.max()
                offMax = gridOffSpikes_filter.max()
                onVsOff[uindex] = (onMax-offMax)/(onMax+offMax)
                
                # SDF time is minLatency before stim onset through 2*maxLatency
                # Hence stim starts at minLatency and analysisWindow starts at 2*minLatency
                # Search analysisWindow for peak but allow searching outside analaysisWindow for halfMax
                sdfMaxInd = np.full((2,3),np.nan)
                preHalfMaxInd = np.full(2,np.nan)
                postHalfMaxInd = np.full(2,np.nan)
                windowOffset = np.where(inAnalysisWindow)[0][0]
                for i,sdf in enumerate((sdfOn,sdfOff)):
                    s = np.copy(sdf)
                    while True:
                        if np.all(np.isnan(s)):
                            break
                        globalMaxInd = list(np.unravel_index(np.nanargmax(s),s.shape))
                        localMaxInd = list(np.unravel_index(np.nanargmax(s[:,:,inAnalysisWindow]),s[:,:,inAnalysisWindow].shape))
                        localMaxInd[2] += windowOffset
                        if globalMaxInd==localMaxInd:
                            sdfMaxInd[i,:] = globalMaxInd
                            break
                        else:
                            s[globalMaxInd[0],globalMaxInd[1],:] = np.nan
                    bestSDF = s[sdfMaxInd[i,0],sdfMaxInd[i,1],:]
                    maxInd = sdfMaxInd[i,2]
                    # respNormArea = (area under SDF in analysisWindow) / (peak * analysisWindow duration)
                    respNormArea[uindex,i] = np.trapz(bestSDF[inAnalysisWindow])*sdfSampInt/(bestSDF[maxInd]*(maxLatency-minLatency))
                    # subtract median(SDF[stimOnset : minLatency])
                    bestSDF -= np.median(bestSDF[np.logical_and(sdfTime>minLatency,sdfTime<minLatency*2)])
                    halfMax = 0.5*bestSDF[maxInd]
                    # find last thresh cross before peak
                    preHalfMaxInd[i] = np.where(bestSDF[:maxInd]<halfMax)[0][-1]+1
                    # find first thresh cross after peak
                    postHalfMax = np.where(bestSDF[maxInd:]<halfMax)[0]
                    postHalfMaxInd[i] = maxInd+postHalfMax[0]-1 if any(postHalfMax) else bestSDF.size-1
                respHalfWidth[uindex,:] = (postHalfMaxInd-preHalfMaxInd)*sdfSampInt
                
#                fwhm = {}
#                indexName = ['on', 'off']                
#                for ind, sdfs in enumerate([sdfOn, sdfOff]):                  
#                    fwhm[indexName[ind]] = {}
#                    maxTrace = np.where(sdfs == sdfs.max())
#                    maxData = sdfs[maxTrace[0][0], maxTrace[1][0], :]
#                    fwhm[indexName[ind]]['range'], fwhm[indexName[ind]]['fw'] = findFWatHM(maxData)
#                    fwhm[indexName[ind]]['trace'] = [maxTrace[0][0], maxTrace[1][0]] 
#                fwOn.append(fwhm['on']['fw'])
#                fwOff.append(fwhm['off']['fw'])                
                                        
                self.units[str(unit)]['rf' + saveTag]['on'] = gridOnSpikes
                self.units[str(unit)]['rf' + saveTag]['off'] = gridOffSpikes
                self.units[str(unit)]['rf' + saveTag]['on_filter'] = gridOnSpikes_filter
                self.units[str(unit)]['rf' + saveTag]['off_filter'] = gridOffSpikes_filter
                self.units[str(unit)]['rf' + saveTag]['xpos'] = xpos
                self.units[str(unit)]['rf' + saveTag]['ypos'] = ypos
                self.units[str(unit)]['rf' + saveTag]['onFit'] = onFit
                self.units[str(unit)]['rf' + saveTag]['offFit'] = offFit
            
            if plot:
                maxVal = max(np.nanmax(gridOnSpikes_filter), np.nanmax(gridOffSpikes_filter))
                minVal = min(np.nanmin(gridOnSpikes_filter), np.nanmin(gridOffSpikes_filter))
                
                sdfMax = max(np.nanmax(sdfOn),np.nanmax(sdfOff))
                spacing = 0.2
                sdfXMax = sdfTime[-1]
                sdfYMax = sdfMax                
                
                for ind,(sdf,resp,fitPrm,title) in enumerate(zip((sdfOn,sdfOff),(gridOnSpikes_filter,gridOffSpikes_filter),(onFit,offFit),('On','Off'))):
                    
                    ax = fig.add_subplot(gs[uindex,ind*3:ind*3+2])
                    x = 0
                    y = 0
                    for i,_ in enumerate(ypos):
                        for j,_ in enumerate(xpos):
                            ax.plot(x+sdfTime,y+sdf[i,j,:],color='k')
                            if not np.isnan(respHalfWidth[uindex,ind]) and i==sdfMaxInd[ind,0] and j==sdfMaxInd[ind,1]:
                                halfMaxInd = [preHalfMaxInd[ind],postHalfMaxInd[ind]]
                                ax.plot(x+sdfTime[halfMaxInd],y+sdf[i,j,halfMaxInd],color='r',linewidth=2)
                            x += sdfXMax*(1+spacing)
                        x = 0
                        y += sdfYMax*(1+spacing)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xticks([minLatency,minLatency+0.1])
                    ax.set_xticklabels(['','100 ms'])
                    ax.set_yticks([0,int(sdfMax)])
                    ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*xpos.size])
                    ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*ypos.size])
                    if ind==0:
                        ax.set_ylabel('Unit '+str(unit)+'\nypos = '+str(round(unitsYPos[uindex])), fontsize='medium')
                    if uindex==0:
                        ax.set_title(title,fontsize='medium')                    
                    
                    ax = fig.add_subplot(gs[uindex,ind*3+2])
                    im = ax.imshow(resp, cmap='jet', clim=(minVal,maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]] )
                    if fit and fitPrm is not None:
                        ax.plot(fitPrm[0],fitPrm[1],'kx',markeredgewidth=2)
                        fitX,fitY = getEllipseXY(*fitPrm[:-1])
                        ax.plot(fitX,fitY,'k',linewidth=2)
                        ax.set_xlim(gridExtent[[0,2]]-0.5)
                        ax.set_ylim(gridExtent[[1,3]]-0.5)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xticks(np.round(azim[[0,-1]]))
                    ax.set_yticks(np.round(elev[[0,-1]]))
                    cb = plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.5, pad=0.04)
                    cb.ax.tick_params(length=0,labelsize='x-small')
                    cb.set_ticks([math.ceil(minVal),int(maxVal)])
                    
        if plot and len(units)>1:
            
            plt.figure(facecolor='w')
            gspec = gridspec.GridSpec(2,2)
            for i,(respDur,ylabel) in enumerate(zip((respNormArea,respHalfWidth),('Norm Area','Half-width'))):
                for j,title in enumerate(('On','Off')):                
                    ax = plt.subplot(gspec[i,j])
                    ax.plot(onVsOff,respDur[:,j],'ko')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([-1,1])
                    ax.set_ylim([0.9*respDur.min(),1.1*respDur.max()])
                    ax.set_xticks([-1,0,1])
                    if i==0:
                        ax.set_xticklabels([])
                        ax.set_title(title,fontsize='small')
                    else:
                        ax.set_xlabel('On vs Off Index',fontsize='small')
                    if j==0:
                        ax.set_ylabel(ylabel,fontsize='small')
                    else:
                        ax.set_yticklabels([])
            
            if fit:
                # comparison of RF and probe position
                plt.figure(facecolor='w')
                unitsYPos = np.array(unitsYPos)
                xlim = np.array([min(unitsYPos)-10,max(unitsYPos)+10])
                # for (azim,elev) in (on,off)...
                for i,rfCenters in enumerate((c[:,xy] for xy in (0,1) for c in (onCenters,offCenters))):
                    ax = plt.subplot(2,2,i+1)
                    self.rfCenters=rfCenters
                    hasRF = np.logical_not(np.isnan(rfCenters))
                    self.hasRF=hasRF
                    self.unitsYPos=unitsYPos
                    if np.count_nonzero(hasRF)>1:
                        # linFit = (slope, intercept, r-value, p-value, stderror)
                        linFit = scipy.stats.linregress(unitsYPos[hasRF],rfCenters[hasRF])
                        ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6')
                        ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(linFit[2]**2,2))+', p = '+str(round(linFit[3],2)),
                                transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
                    ax.plot(unitsYPos,rfCenters,'ko')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                    ax.set_xlim(xlim)
                    if i==0:
                        ax.set_ylabel('Azimuth',fontsize='medium')
                        ax.set_title('On',fontsize='medium')
                    elif i==1:
                        ax.set_title('Off',fontsize='medium')
                    elif i==2:
                        ax.set_ylabel('Elevation',fontsize='medium')
                    if i in (0,1):
                        ax.set_ylim(gridExtent[[0,2]]+[-5,5])
                    else:
                        ax.set_ylim(gridExtent[[1,3]]+[-5,5])
                        ax.set_xlabel('Probe Y Pos',fontsize='medium')
            
    
    def analyzeGratings(self, units=None, trials = None, responseLatency = 0.25, plot=True, protocol=None, protocolType='stf', avgOri = True, fit = True, useCache=False, saveTag=''):
    
        units, unitsYPos = self.getOrderedUnits(units) 
            
        if protocol is None:
            if protocolType=='stf':
                label = 'gratings'
            elif protocolType=='ori':
                label = 'gratings_ori'
            protocol = self.getProtocolIndex(label)
        protocol = str(protocol)
               
        if plot:
            plt.figure(figsize =(10, 4*len(units)),facecolor='w')
            if protocolType=='stf':
                gridWidth = 4 if avgOri else 8
            else:
                gridWidth = 1
            gs = gridspec.GridSpec(len(units), gridWidth)

        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
        
        latencySamples = int(responseLatency*self.sampleRate)
        
        # ignore trials with bogus sf or tf
        trialContrast = self.visstimData[str(protocol)]['stimulusHistory_contrast'][trials]
        trialSF = self.visstimData[str(protocol)]['stimulusHistory_sf'][trials]
        trialTF = self.visstimData[str(protocol)]['stimulusHistory_tf'][trials]
        tol = 1e-9
        sf = np.copy(self.visstimData[protocol]['sf'])
        sf = sf[np.logical_and(sf>0.01-tol,sf<0.32+tol)]
        tf = np.copy(self.visstimData[protocol]['tf'])
        tf = tf[np.logical_and(tf>0.5-tol,tf<8+tol)]
        stfTrialsToUse = np.logical_and(np.logical_and(trialSF>0.01-tol,trialSF<0.32+tol),
                                        np.logical_and(trialTF>0.5-tol,trialTF<8+tol))
        trialsToUse = np.logical_or(np.isclose(trialContrast,0),stfTrialsToUse)
        trials = trials[trialsToUse]         
        
        trialContrast = trialContrast[trialsToUse]
        trialSF = trialSF[trialsToUse]
        trialTF = trialTF[trialsToUse]
        
        # as presented ori 0 is rightward moving vertical bars and ori 90 is downward moving horizontal bars
        # change to more typical convention (90 up)
        trialOri = np.copy(self.visstimData[str(protocol)]['stimulusHistory_ori'][trials])
        trialOri[trialOri>0] = -(trialOri[trialOri>0]-360)
        ori = np.copy(self.visstimData[protocol]['ori'])
        ori[ori>0] = -(ori[ori>0]-360)
        ori = np.sort(ori)  
        
        stfOri = ['avg'] if avgOri else ori
        
        trialStartFrame = self.visstimData[protocol]['stimStartFrames'][trials]
        trialDuration = self.visstimData[protocol]['stimTime']
        trialStartSamples = self.visstimData[protocol]['frameSamples'][trialStartFrame] + latencySamples
        trialEndSamples = self.visstimData[protocol]['frameSamples'][trialStartFrame+trialDuration]
        
        preTime = self.visstimData[protocol]['preTime']/self.visstimData[protocol]['frameRate']
        stimTime = self.visstimData[protocol]['stimTime']/self.visstimData[protocol]['frameRate']
        postTime = self.visstimData[protocol]['postTime']/self.visstimData[protocol]['frameRate']
        sdfSamples = round((preTime+stimTime+postTime)*self.sampleRate)
        sdfSigma = 0.02
        sdfSampInt = 0.001
        
        for uindex, unit in enumerate(units):
            if ('stf' + saveTag) in self.units[str(unit)] and protocolType=='stf' and useCache:
                stfMat = self.units[str(unit)]['stf' + saveTag]['stfMat']
                sf = self.units[str(unit)]['stf' + saveTag]['sf']
                tf = self.units[str(unit)]['stf' + saveTag]['tf']
                fitParams = self.units[str(unit)]['stf' + saveTag]['fitParams']
            elif ('ori' + saveTag) in self.units[str(unit)] and protocolType=='ori' and useCache:
                oriList = self.units[str(unit)]['ori' + saveTag]['tuningCurve']
            else:
                self.units[str(unit)][protocolType + saveTag] = {}
                
                #spontaneous firing rate taken from interspersed gray trials
                spontRate = 0
                spontCount = 0
                
                stfMat = np.zeros((tf.size,sf.size,ori.size))
                stfCountMat = np.zeros_like(stfMat)
                oriList = [[] for i in range(ori.size)]
                
                spikes = self.units[str(unit)]['times'][protocol]
                spikesPerTrial = self.findSpikesPerTrial(trialStartSamples,trialEndSamples,spikes)
                trialResponse = spikesPerTrial/((trialEndSamples-trialStartSamples)/self.sampleRate)
                
                #make STF mat for specified trials (default all trials)
                for trial,_ in enumerate(trials):
                    spikeRateThisTrial = trialResponse[trial]
                    if trialContrast[trial] > 0+tol:
                        sfIndex = np.where(np.isclose(sf, trialSF[trial]))[0]
                        tfIndex = np.where(np.isclose(tf, trialTF[trial]))[0]
                        oriIndex = np.where(np.isclose(ori,trialOri[trial]))[0]
                        stfMat[tfIndex, sfIndex, oriIndex] += spikeRateThisTrial
                        stfCountMat[tfIndex, sfIndex, oriIndex] += 1
                        oriList[oriIndex].append(spikeRateThisTrial)
                    else:
                        spontRate += spikeRateThisTrial
                        spontCount += 1
                
                if spontCount>0:
                    spontRate /= spontCount
                else:
                    spontRate = np.nan
                    
                self.units[str(unit)][protocolType + saveTag]['spontRate'] = spontRate
                
                stfCountMat[stfCountMat==0] = np.nan
                stfMat /= stfCountMat
                if avgOri:
                    stfMat = np.nanmean(stfMat,axis=2)
                
                oriMean = np.zeros(len(oriList))                
                oriError = np.zeros(len(oriList))
                for oindex in range(len(oriList)):
                    oriMean[oindex] = np.mean(np.array(oriList[oindex]))
                    oriError[oindex] = np.std(np.array(oriList[oindex]))
                
                sdf = np.full((tf.size,sf.size,round(sdfSamples/self.sampleRate/sdfSampInt),len(stfOri)),np.nan)
                f1Mat = np.full((tf.size,sf.size,len(stfOri)),np.nan)
                contrastTrials = trialContrast>0+tol
                for tfInd,thisTF in enumerate(tf):
                    tfTrials = np.isclose(trialTF,thisTF)
                    for sfInd,thisSF in enumerate(sf):
                        sfTrials = np.isclose(trialSF,thisSF)
                        for oriInd,thisOri in enumerate(stfOri):
                            if avgOri:
                                trialIndex = np.ones(trials.size,dtype=bool)
                            else:
                                trialIndex = np.isclose(trialOri,thisOri)
                            for i in (contrastTrials,tfTrials,sfTrials):
                                trialIndex = np.logical_and(trialIndex,i)
                            if any(trialIndex):
                                sdf[tfInd,sfInd,:,oriInd],sdfTime = self.getMeanSDF(spikes,trialStartSamples[trialIndex]-latencySamples-int(preTime*self.sampleRate),sdfSamples,sigma=sdfSigma,sampInt=sdfSampInt)
                                inAnalysisWindow = np.logical_and(sdfTime>preTime+responseLatency,sdfTime<preTime+stimTime)
                                s = sdf[tfInd,sfInd,inAnalysisWindow,oriInd]
                                f,pwr = scipy.signal.welch(s,1/sdfTime[1],nperseg=s.size,detrend='constant',scaling='spectrum')
                                pwr **= 0.5
                                f1Ind = np.argmin(np.absolute(f-thisTF))
                                f1Mat[tfInd,sfInd,oriInd] = pwr[f1Ind-1:f1Ind+2].max()/s.mean()
                if avgOri:
                    sdf = np.squeeze(sdf)
                    f1Mat = np.squeeze(f1Mat)
                
                if fit and protocolType=='stf':
                    # params: sf0 , tf0, sigSF, sigTF, speedTuningIndex, amplitude
                    fitParams = []
                    for oriInd,_ in enumerate(stfOri):
                        stfMatOri = np.copy(stfMat) if avgOri else np.copy(stfMat[:,:,oriInd])
                        stfMatOri -= spontRate
                        if np.any(np.isnan(stfMatOri)) or stfMatOri.max()<0:
                            fitParams.append(None)
                        else:
                            i,j = np.unravel_index(np.argmax(stfMatOri),stfMatOri.shape)
                            initialParams = (sf[j], tf[i], 1, 1, 0.5, stfMatOri.max())
                            fitParams.append(fitStfLogGauss2D(sf,tf,stfMatOri,initialParams))   
                    self.units[str(unit)]['stf' + saveTag]['fitParams'] = fitParams
                    
                if protocolType=='stf':
                    self.units[str(unit)]['stf' + saveTag]['stfMat'] = stfMat
                    self.units[str(unit)]['stf' + saveTag]['sf'] = sf
                    self.units[str(unit)]['stf' + saveTag]['tf'] = tf
                    
                if protocolType=='ori':
                    dsi,prefDir = getDSI(oriMean,ori)
                    osi,prefOri = getDSI(oriMean,2*ori)
                    prefOri /= 2
                    self.units[str(unit)]['ori' + saveTag]['dsi'] = [dsi, prefDir]
                    self.units[str(unit)]['ori' + saveTag]['osi'] = [osi, prefOri]
                    self.units[str(unit)]['ori' + saveTag]['tuningCurve'] = [oriMean, oriError]
        
            if plot:
                if protocolType=='stf':
                    for oriInd,thisOri in enumerate(stfOri):
                        ax = plt.subplot(gs[uindex,oriInd*4:oriInd*4+2])
                        sdfOri = sdf if avgOri else sdf[:,:,:,oriInd]
                        spacing = 0.2
                        sdfXMax = sdfTime[-1]
                        sdfYMax = np.nanmax(sdfOri)
                        x = 0
                        y = 0
                        for i,_ in enumerate(tf):
                            for j,_ in enumerate(sf):
                                ax.plot(x+sdfTime,y+sdfOri[i,j,:],color='k')
                                x += sdfXMax*(1+spacing)
                            x = 0
                            y += sdfYMax*(1+spacing)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([preTime,preTime+stimTime])
                        ax.set_xticklabels(['0',str(stimTime)+' s'])
                        ax.set_yticks([0,int(sdfYMax)])
                        ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*sf.size])
                        ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*tf.size])
                        if oriInd==0:
                            ax.set_ylabel('Unit '+str(unit), fontsize='medium')
                        if uindex==0:
                            ax.set_title('ori = '+str(thisOri),fontsize='medium') 
                       
                        ax = plt.subplot(gs[uindex,oriInd*4+2])
                        stfMatOri = np.copy(stfMat) if avgOri else np.copy(stfMat[:,:,oriInd])
                        f1MatOri = np.copy(f1Mat) if avgOri else np.copy(f1Mat[:,:,oriInd])
                        xyNan = np.transpose(np.where(np.isnan(stfMatOri)))
                        nanTrials = np.isnan(stfMatOri)
                        stfMatOri[nanTrials] = 0
                        f1MatOri[nanTrials] = 0
                        centerPoint = spontRate if not np.isnan(spontRate) else np.nanmedian(stfMatOri)
                        cLim = np.nanmax(abs(stfMatOri-centerPoint))
                        im = ax.imshow(stfMatOri, clim=(centerPoint-cLim, centerPoint+cLim), cmap='bwr', origin = 'lower', interpolation='none')
                        for xypair in xyNan:    
                            ax.text(xypair[1], xypair[0], 'nan', color='white', ha='center')
                        if fit and fitParams[oriInd] is not None:
                            ax.plot(np.log2(fitParams[oriInd][0])-np.log2(sf[0]),np.log2(fitParams[oriInd][1])-np.log2(tf[0]),'kx',markeredgewidth=2)
                            fitX,fitY = getStfContour(sf,tf,fitParams[oriInd])
                            ax.plot(fitX,fitY,'k',linewidth=2)
                            ax.set_xlim([-0.5,sf.size-0.5])
                            ax.set_ylim([-0.5,tf.size-0.5])
                        ax.tick_params(labelsize='x-small')
                        ax.set_xticks(range(sf.size))
                        ax.set_yticks(range(tf.size))
                        ax.set_xticklabels(sf)
                        ax.set_yticklabels(tf)
                        ax.set_xlabel('Cycles/deg',fontsize='small')
                        ax.set_ylabel('Cycles/s',fontsize='small')
                        if uindex==0:
                            ax.set_title('Spikes/s',fontsize='small')
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                        
                        ax = plt.subplot(gs[uindex,oriInd*4+3])
                        im = ax.imshow(f1MatOri, clim=(0,f1MatOri.max()), cmap='gray', origin = 'lower', interpolation='none')
                        ax.tick_params(labelsize='x-small')
                        ax.set_xticks(range(sf.size))
                        ax.set_yticks(range(tf.size))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        if uindex==0:
                            ax.set_title('f1/f0',fontsize='small')
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([0,round(int(f1MatOri.max()*100)/100,2)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                    
#                    a2 = plt.subplot(gs[uindex,2])
#                    values = np.mean(stfMat, axis=0)
#                    error = np.std(stfMat, axis=0)
#                    a2.plot(sf, values)
#                    plt.fill_between(sf, values+error, values-error, alpha=0.3)
#                    plt.xlabel('sf')
#                    plt.ylabel('spikes')
#                    plt.xticks(sf)
#                    
#                    a3 = plt.subplot(gs[uindex,3])
#                    values = np.mean(stfMat, axis=1)
#                    error = np.std(stfMat, axis=1)
#                    a3.plot(tf, values)
#                    plt.fill_between(tf, values+error, values-error, alpha=0.3)
#                    plt.xlabel('tf')
#                    plt.ylabel('spikes')
#                    plt.xticks(tf)
              
                elif protocolType=='ori':
                    a1 = plt.subplot(gs[uindex,0], projection='polar')
                    theta = ori * (np.pi/180.0)
                    theta = np.append(theta, theta[0])
                    rho = np.append(oriMean, oriMean[0])
                    a1.plot(theta, rho)
                    a1.set_title('DSI = '+str(round(dsi,2))+', prefDir = '+str(round(prefDir))+'\n'+'OSI = '+str(round(osi,2))+', prefOri = '+str(round(prefOri)),fontsize='x-small')
                    
#                    a2 = plt.subplot(gs[uindex,1])
#                    plt.xlabel('ori')
#                    plt.ylabel('spike rate: ' + str(unit))
#                    a2.plot(ori, oriMean)
#                    plt.fill_between(ori, oriMean+oriError, oriMean - oriError, alpha=0.3)
#                    plt.xticks(ori)


    def analyzeSpots(self, units=None, protocol = None, plot=True, trials=None, useCache=False, saveTag=''):
         
        units, unitsYPos = self.getOrderedUnits(units) 
         
        if protocol is None:
            protocol = self.getProtocolIndex('spots')
        protocol = str(protocol)
         
        if plot:        
            plt.figure(figsize = (10, 4*len(units)))
            gs = gridspec.GridSpec(2*len(units), 4)                        
         
        if trials is None:
            trials = np.arange((self.visstimData[str(protocol)]['trialStartFrame'][:-1]).size)
         
        trialStartFrames = self.visstimData[str(protocol)]['trialStartFrame'][trials]
        trialDuration = (self.visstimData[str(protocol)]['trialNumFrames'][trials]).astype(np.int)
        trialEndFrames = trialStartFrames + trialDuration
        frameSamples = self.visstimData[str(protocol)]['frameSamples']     
        trialStartSamples = frameSamples[trialStartFrames]
        trialEndSamples = frameSamples[trialEndFrames]
         
        trialPos = self.visstimData[str(protocol)]['trialSpotPos'][trials]
        trialColor = self.visstimData[str(protocol)]['trialSpotColor'][trials]
        trialSize = self.visstimData[str(protocol)]['trialSpotSize'][trials]
        trialDir = self.visstimData[str(protocol)]['trialSpotDir'][trials]
        trialSpeed = self.visstimData[str(protocol)]['trialSpotSpeed'][trials]
        spotPos = np.unique(trialPos)
        spotColor = np.unique(trialColor)
        spotSize = np.unique(trialSize)
        spotDir = np.unique(trialDir)
        spotSpeed = np.unique(trialSpeed)
   
        horzTrials = np.logical_or(trialDir==0, trialDir==180)
        vertTrials = np.logical_or(trialDir==270, trialDir==90)
        azimuths = np.unique(trialPos[vertTrials])
        elevs = np.unique(trialPos[horzTrials])
        
        numTrialTypes = spotSpeed.size*spotSize.size*(2*azimuths.size+elevs.size)*spotColor.size
        maxTrialsPerType = math.ceil(trials.size/numTrialTypes)
        resp = np.full((spotSpeed.size,spotSize.size,spotDir.size,spotPos.size,spotColor.size,maxTrialsPerType),np.nan)
         
        for uindex, unit in enumerate(units):
            if ('spotResponse' + saveTag) in self.units[str(unit)] and useCache:
                responseDict = self.units[str(unit)]['spotResponse' + saveTag]['spot_responseDict']
                spotRF = responseDict['spotRF']
                spontRate = responseDict['spontRate']
            else:
                self.units[str(unit)]['spotResponse' + saveTag] = {}
                spikes = self.units[str(unit)]['times'][str(protocol)]
         
                # get RF         
                spikesPerTrial = self.findSpikesPerTrial(trialStartSamples, trialEndSamples, spikes)
                trialSpikeRate = spikesPerTrial/((1/self.visstimData[str(protocol)]['frameRate'])*trialDuration)
 
                azimuthSpikeRate = np.zeros(azimuths.size)        
                elevSpikeRate = np.zeros(elevs.size)
                azimuthTrialCount = np.zeros(azimuths.size)        
                elevTrialCount = np.zeros(elevs.size)
                for trial in range(trialPos.size):
                    if horzTrials[trial]:
                        elevIndex = np.where(trialPos[trial]==elevs)[0]
                        elevSpikeRate[elevIndex] += trialSpikeRate[trial]
                        elevTrialCount[elevIndex] += 1
                    else:
                        azimuthIndex = np.where(trialPos[trial]==azimuths)[0]
                        azimuthSpikeRate[azimuthIndex] += trialSpikeRate[trial]
                        azimuthTrialCount[azimuthIndex] += 1
                 
                elevSpikeRate /= elevTrialCount
                azimuthSpikeRate /= azimuthTrialCount
         
                #get spontaneous rate
                recoveryPeriod = 10
                interTrialIntervals = trialStartFrames[1:]- trialEndFrames[:-1]
                interTrialStarts = trialEndFrames[:-1] + recoveryPeriod
                interTrialEnds = trialEndFrames[:-1] + interTrialIntervals        
                itiSpikes = self.findSpikesPerTrial(frameSamples[interTrialStarts], frameSamples[interTrialEnds], spikes)
                itiRate = itiSpikes/((1/60.0)*(interTrialEnds - interTrialStarts))
                spontRate = itiRate.mean()
                sdfSigma = 0.1
                sdf,_ = self.getMeanSDF(spikes,frameSamples[interTrialStarts],max(frameSamples[interTrialEnds]-frameSamples[interTrialStarts]),sigma=sdfSigma)
                peakSpontRate = sdf.max()
                 
                #make tuning curves for various spot parameters        
                responseDict = {}        
                for param in ['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']:
                    trialValues = self.visstimData[str(protocol)][param][trials]            
                    possibleValues = np.unique(trialValues)
                    responseDict[param] = {}
                    meanResponse = np.zeros(possibleValues.size)
                    semResponse = np.zeros(possibleValues.size)
                    for ind, value in enumerate(possibleValues):
                        relevantTrials = np.where(trialValues==value)[0]
                        responseDict[param][value] = {}
                        responseDict[param][value]['trials'] = relevantTrials
                        responseDict[param][value]['response'] = np.zeros(relevantTrials.size)
                        for index, trial in enumerate(relevantTrials):
                            totalSpikes = spikesPerTrial[trial]
                            spikeRate = totalSpikes/((1/60.0)*trialDuration[trial])            
                            responseDict[param][value]['response'][index] = spikeRate
                        meanResponse[ind] = np.mean(responseDict[param][value]['response'])
                        semResponse[ind] = np.std(responseDict[param][value]['response'])/math.sqrt(relevantTrials.size)
                    responseDict[param]['tuningCurve'] = {}
                    responseDict[param]['tuningCurve']['paramValues'] = possibleValues
                    responseDict[param]['tuningCurve']['meanResponse'] = meanResponse
                    responseDict[param]['tuningCurve']['sem'] = semResponse                     
                     
                x,y = np.meshgrid(azimuthSpikeRate,elevSpikeRate)
                spotRF = np.sqrt(abs(x*y))*np.sign(x+y)
                responseDict['spontRate'] = spontRate
                responseDict['spotRF'] = spotRF                
                self.units[str(unit)]['spotResponse' + saveTag]['spot_responseDict'] = responseDict
                
                # speed x size x dir x pos x color matrix for mean and peak resp
                peakResp = np.full(resp.shape[:-1],np.nan)
                for speedInd,speed in enumerate(spotSpeed):
                    speedTrials = trialSpeed==speed
                    for sizeInd,size in enumerate(spotSize):
                        sizeTrials = trialSize==size
                        for dirInd,direction in enumerate(spotDir):
                            dirTrials = trialDir==direction
                            for posInd,pos in enumerate(spotPos):
                                posTrials = trialPos==pos
                                for colorInd,color in enumerate(spotColor):
                                    trialInd = trialColor==color
                                    for i in (speedTrials,sizeTrials,dirTrials,posTrials):
                                        trialInd = np.logical_and(trialInd,i)
                                    if any(trialInd):
                                        resp[speedInd,sizeInd,dirInd,posInd,colorInd,:np.count_nonzero(trialInd)] = trialSpikeRate[trialInd]
                                        sdf,_ = self.getMeanSDF(spikes,trialStartSamples[trialInd],max(trialEndSamples[trialInd]-trialStartSamples[trialInd]),sigma=sdfSigma)
                                        peakResp[speedInd,sizeInd,dirInd,posInd,colorInd] = sdf.max()
                meanResp = np.nanmean(resp,axis=5)
                resp[:] = np.nan
                 
            if plot:   
#                a1 = plt.subplot(gs[uindex, 0]) 
#                centerPoint = spontRate if not np.isnan(spontRate) else np.nanmedian(spotRF)
#                cLim = np.nanmax(abs(spotRF-centerPoint))
#                im = a1.imshow(spotRF, clim=(centerPoint-cLim,centerPoint+cLim), cmap='bwr', interpolation='none', origin='lower')
#                plt.colorbar(im, ax=a1, fraction=0.05, pad=0.04)
#                plt.title(str(unit), fontsize='x-small')
                 
#                for paramnum, param in enumerate(['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']):
#                        a = plt.subplot(gs[uindex, paramnum+1])
#                        values = responseDict[param]['tuningCurve']['mean_spontSubtracted'] 
#                        error = responseDict[param]['tuningCurve']['sem'] 
#                        a.plot(responseDict[param]['tuningCurve']['paramValues'], values)
#                        plt.fill_between(responseDict[param]['tuningCurve']['paramValues'], values+error, values-error, alpha=0.3)
#                        a.plot(responseDict[param]['tuningCurve']['paramValues'], np.zeros(values.size), 'r--')
#                        plt.xlabel(param) 
#                        plt.ylim(min(-0.1, np.min(values - error)), max(np.max(values + error), 0.1))
#                        plt.locator_params(axis = 'y', nbins = 3)
#                        a.set_xticks(responseDict[param]['tuningCurve']['paramValues'])
                 
                 axInd = 0
                 for r,spRate in zip((meanResp,peakResp),(spontRate,peakSpontRate)):
                    for m in ('mean','max'):
                        # speed vs size
                        ax = plt.subplot(gs[uindex*2,axInd])
                        if m=='mean':
                            speedSizeResp = np.nanmean(np.nanmean(np.nanmean(r,axis=4),axis=3),axis=2)
                        else:
                            speedSizeResp = np.nanmax(np.nanmax(np.nanmax(r,axis=4),axis=3),axis=2)
                        centerPoint = spRate if not np.isnan(spRate) else np.nanmedian(speedSizeResp)
                        cLim = np.nanmax(abs(speedSizeResp-centerPoint))
                        plt.imshow(speedSizeResp,cmap='bwr',clim=(centerPoint-cLim,centerPoint+cLim),interpolation='none',origin='lower')
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xticks(range(spotSize.size))
                        ax.set_xticklabels([])
                        ax.set_yticks(range(spotSpeed.size))
                        ax.set_yticklabels([])
                        if axInd==0:
                            ax.set_yticklabels(spotSpeed)
                            ylab = 'Unit '+str(unit)+'\nSpot Speed' if uindex==0 else 'Unit '+str(unit)
                            ax.set_ylabel(ylab,fontsize='x-small')
                        if uindex==len(units)-1 and axInd==0:
                            ax.set_xticklabels(spotSize)
                            ax.set_xlabel('Spot Size',fontsize='x-small')
                        if uindex==0:
                            if axInd==0:
                                title = 'meanResp\nmean'
                            elif axInd==1:
                                title = 'meanResp\nmax'
                            elif axInd==2:
                                title = 'peakResp\nmean'
                            else:
                                title = 'peakResp\nmax'
                            ax.set_title(title,fontsize='x-small')
                        cb = plt.colorbar(fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                        
                        # direction
                        ax = plt.subplot(gs[uindex*2+1,axInd])
                        if m=='mean':
                            dirResp = np.nanmean(np.nanmean(np.nanmean(np.nanmean(r,axis=4),axis=3),axis=1),axis=0)
                        else:
                            dirResp = np.nanmax(np.nanmax(np.nanmax(np.nanmax(r,axis=4),axis=3),axis=1),axis=0)
                        plt.plot(spotDir,dirResp)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xticks(spotDir)
                        ax.set_xticklabels([])
                        ax.set_xlim([spotDir[0],spotDir[-1]])
                        ax.set_ylim([0,dirResp.max()+0.5])
                        if axInd==0:
                            ylab = 'Unit '+str(unit)+'\nSpikes/s' if uindex==0 else 'Unit '+str(unit)
                            ax.set_ylabel(ylab,fontsize='x-small')
                        if uindex==len(units)-1 and axInd==0:
                            ax.set_xlabel('Direction',fontsize='x-small')
                            ax.set_xticklabels(spotDir)
                        
                        axInd += 1
    
                                        
    def analyzeCheckerboard(self, units=None, protocol=None, trials=None, plot=True, saveTag=''):
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('checkerboard')
        protocol = str(protocol)          
        p = self.visstimData[protocol]
        assert(set(p['bckgndDir'])=={0,180} and set(p['patchDir'])=={0,180} and 0 in p['bckgndSpeed'] and 0 in p['patchSpeed'])
        
        if trials is None:
            trials = np.arange((p['trialStartFrame']).size)
        if p['frameSamples'][p['trialStartFrame'][-1]]+p['trialNumFrames'][-1]/p['frameRate']*self.sampleRate>p['frameSamples'][-1]:
            trials = trials[:-1]   
        trialStartFrame = p['trialStartFrame'][trials]
        trialNumFrames = (p['trialNumFrames'][trials]).astype(int)
        trialStartSamples = p['frameSamples'][trialStartFrame]
        trialEndSamples = p['frameSamples'][trialStartFrame+trialNumFrames]
        minInterTrialTime = p['interTrialInterval'][0]
        minInterTrialSamples = int(minInterTrialTime*self.sampleRate)
        
        if plot:
            plt.figure(figsize=(10,4*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),3)
        
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        numTrialTypes = patchSpeed.size*bckgndSpeed.size*p['patchSize'].size*p['patchElevation'].size
        numTrialTypes -= (bckgndSpeed.size+patchSpeed.size-1)*p['patchSize'].size*p['patchElevation'].size
        maxTrialsPerType = math.ceil(trials.size/numTrialTypes)
        resp = np.full((patchSpeed.size,bckgndSpeed.size,p['patchSize'].size,p['patchElevation'].size,maxTrialsPerType),np.nan)
        sdfSigma = 0.1
        sdfSampInt = 0.001
        sdfTime = np.arange(0,2*minInterTrialTime+max(trialEndSamples-trialStartSamples)/self.sampleRate,sdfSampInt)
        for uInd,u in enumerate(units):
            spikes = self.units[str(u)]['times'][protocol]
            
            # get mean spikes/s and spontRate
            spikesPerTrial = self.findSpikesPerTrial(trialStartSamples,trialEndSamples,spikes)
            trialSpikeRate = spikesPerTrial/((1/p['frameRate'])*trialNumFrames)
            for i,trial in enumerate(trials):
                pchSpeedInd = patchSpeed==p['trialPatchSpeed'][trial]if p['trialPatchDir'][trial]==0 else patchSpeed==-p['trialPatchSpeed'][trial]
                bckSpeedInd = bckgndSpeed==p['trialBckgndSpeed'][trial] if p['trialBckgndDir'][trial]==0 else bckgndSpeed==-p['trialBckgndSpeed'][trial]
                pchSizeInd = p['patchSize']==p['trialPatchSize'][trial]
                pchElevInd = p['patchElevation']==p['trialPatchPos'][trial]
                n = np.count_nonzero(np.logical_not(np.isnan(resp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd,:])))
                resp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd,n] = trialSpikeRate[i]
            meanResp = np.nanmean(resp,axis=4)
            meanSpontRate = np.nanmean(resp[patchSpeed.size//2,bckgndSpeed.size//2,:,:,:])
            resp[:] = np.nan
            
            # get sdfs and peak spikes/s
            sdf = np.full(np.concatenate((meanResp.shape,[sdfTime.size])),np.nan)
            peakResp = np.full(meanResp.shape,np.nan)
            for pchSpeedInd,pchSpeed in enumerate(patchSpeed):
                pchDir = 180 if pchSpeed<0 else 0
                a = p['trialPatchDir'][trials]==pchDir
                b = p['trialPatchSpeed'][trials]==abs(pchSpeed)
                for bckSpeedInd,bckSpeed in enumerate(bckgndSpeed):
                    bckDir = 180 if bckSpeed<0 else 0
                    c = p['trialBckgndDir'][trials]==bckDir
                    d = p['trialBckgndSpeed'][trials]==abs(bckSpeed)
                    for pchSizeInd,pchSize in enumerate(p['patchSize']):
                        e = p['trialPatchSize'][trials]==pchSize
                        for pchElevInd,pchElev in enumerate(p['patchElevation']):
                            trialInd = p['trialPatchPos'][trials]==pchElev
                            for i in (a,b,c,d,e):
                                trialInd = np.logical_and(trialInd,i)
                            if any(trialInd):
                                s,_ = self.getMeanSDF(spikes,trialStartSamples[trialInd]-minInterTrialSamples,2*minInterTrialSamples+max(trialEndSamples[trialInd]-trialStartSamples[trialInd]),sigma=sdfSigma,sampInt=sdfSampInt)
                                peakResp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd] = s.max()
                                sdf[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd,:s.size] = s
            peakSpontRate = np.nanmean(peakResp[patchSpeed.size//2,bckgndSpeed.size//2,:,:])
            
            # fill in resp for patch and bckgnd speeds not tested for every patch size and elevation
            for r in (meanResp,peakResp):
                for pchSizeInd,_ in enumerate(p['patchSize']):
                    for pchElevInd,_ in enumerate(p['patchElevation']):
                        r[patchSpeed.size//2,:,pchSizeInd,pchElevInd] = r[patchSpeed.size//2,:,0,0]
                for pchSpeedInd,pchSpeed in enumerate(patchSpeed):
                    for bckSpeedInd,bckSpeed in enumerate(bckgndSpeed):
                        if pchSpeed==bckSpeed:
                            r[pchSpeedInd,bckSpeedInd,:,:] = r[patchSpeed.size//2,bckSpeedInd]            
            
            self.units[str(u)]['checkerboard' + saveTag] = {'meanResp':meanResp,
                                                            'meanSpontRate':meanSpontRate,
                                                            'peakResp':peakResp,
                                                            'peakSpontRate':peakSpontRate}
            
            if plot:
                ax = plt.subplot(gs[uInd,0:2])
                spacing = 0.2
                sdfXMax = sdfTime[-1]
                sdfYMax = np.nanmax(sdf)
                x = 0
                y = 0
                for i,_ in enumerate(patchSpeed):
                    for j,_ in enumerate(bckgndSpeed):
                        sizeInd,elevInd = np.unravel_index(np.argmax(meanResp[i,j,:,:]),meanResp[i,j,:,:].shape)
                        ax.plot(x+sdfTime,y+sdf[i,j,sizeInd,elevInd,:],color='k')
                        x += sdfXMax*(1+spacing)
                    x = 0
                    y += sdfYMax*(1+spacing)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xticks([minInterTrialTime,sdfTime[-1]-2*minInterTrialTime])
                ax.set_xticklabels(['0',str(int(sdfTime[-1]-2*minInterTrialTime))+' s'])
                ax.set_yticks([0,int(sdfYMax)])
                ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*bckgndSpeed.size])
                ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*patchSpeed.size])
                ax.set_ylabel('Unit '+str(u), fontsize='medium')
                
                ax = plt.subplot(gs[uInd,2])
                respMat = np.nanmax(np.nanmax(meanResp,axis=3),axis=2) # max resp over all patch sizes and elevations
                centerPoint = meanSpontRate if not np.isnan(meanSpontRate) else np.nanmedian(respMat)
                cLim = np.nanmax(abs(respMat-centerPoint))
                plt.imshow(respMat,cmap='bwr',clim=(centerPoint-cLim,centerPoint+cLim),interpolation='none',origin='lower')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xticks(range(bckgndSpeed.size))
                ax.set_xticklabels(bckgndSpeed)
                ax.set_yticks(range(patchSpeed.size))
                ax.set_yticklabels(patchSpeed)
                ax.set_xlabel('Background Speed')
                ax.set_ylabel('Patch Speed')
                ax.set_title('Spikes/s',fontsize='small')
                cb = plt.colorbar(fraction=0.05,pad=0.04,shrink=0.5)
                cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                cb.ax.tick_params(length=0,labelsize='xx-small')
    
    
    def parseRunning(self, protocol, runThresh = 5.0, statThresh = 1.0, trialStarts = None, trialEnds = None, wheelDownsampleFactor = 500.0):
        if not 'running' in self.behaviorData[str(protocol)]:
            self.decodeWheel(self.d[protocol]['data'][::500, self.wheelChannel]*self.d[protocol]['gains'][self.wheelChannel])
        
        wheelData = -self.behaviorData[str(protocol)]['running']
        
        if trialStarts is not None:
            runningTrials = []
            stationaryTrials = []
            for trial in range(trialStarts.size):
                trialSpeed = np.mean(wheelData[round(trialStarts[trial]/wheelDownsampleFactor):round(trialEnds[trial]/wheelDownsampleFactor)])
                if trialSpeed >= runThresh:
                    runningTrials.append(trial)
                elif trialSpeed <= statThresh:
                    stationaryTrials.append(trial)
        return stationaryTrials, runningTrials, trialSpeed
    
    
    def analyzeRunning(self, units, protocol, plot=True):
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if plot:
            plt.figure(figsize=(10,3*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),1)
                        
        kernelWidth=500.0
        for uindex, u in enumerate(units):
            spikes = self.units[str(u)]['times'][str(protocol)]
            wd = -self.behaviorData[str(protocol)]['running']
            fh, _ = np.histogram(spikes, np.arange(0, (wd.size+1)*int(kernelWidth), int(kernelWidth)))
            frc = np.convolve(fh, np.ones(kernelWidth),'same')/kernelWidth
            
            speedBins = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            binnedSpeed = np.digitize(wd, speedBins)
            fr_binned = []
            fr_std = []
            for sbin, _ in enumerate(speedBins):
                binIndices = binnedSpeed==sbin
                #print binIndices
                fr_thisbin = np.mean(frc[binIndices])
                fr_stdThisBin = np.std(frc[binIndices])
                fr_binned.append(fr_thisbin)
                fr_std.append(fr_stdThisBin)
            
            fr_binned = np.array(fr_binned)
            fr_std = np.array(fr_std)
            self.units[str(u)]['runModulation'] = {}
            self.units[str(u)]['runModulation'][str(protocol)] = [speedBins, fr_binned, fr_std] 
            
            if plot:
                ax = plt.subplot(gs[uindex, 0])
                ax.plot(speedBins, fr_binned)
#                ax.set_xscale('log', basex=2)
                plt.fill_between(speedBins, fr_binned+fr_std, fr_binned-fr_std, alpha=0.3)
    
    
    def plotISIHist(self,units=None,protocol=None,binWidth=0.001,maxInterval=0.02):
        units,unitsYPos = self.getOrderedUnits(units)
        if protocol is None:
            protocol = range(len(self.kwdFileList))
        elif not isinstance(protocol,list):
            protocol = [protocol]
        bins = np.arange(0,maxInterval+binWidth,binWidth)
        plt.figure(facecolor='w')
        gs = gridspec.GridSpec(len(units),len(protocol))
        for i,u in enumerate(units):
            ax = []
            ymax = 0
            for j,p in enumerate(protocol):
                spikeTimes = self.units[u]['times'][str(p)]/self.sampleRate
                isiHist,_ = np.histogram(np.diff(spikeTimes),bins)
                isiProb = isiHist/spikeTimes.size
                ymax = max(ymax,isiProb.max())
                ax.append(plt.subplot(gs[i,j]))
                ax[-1].bar(bins[:-1],isiProb,binWidth,color='b',edgecolor='b')
                ax[-1].set_xlim([0,maxInterval])
                ax[-1].spines['right'].set_visible(False)
                ax[-1].spines['top'].set_visible(False)
                ax[-1].tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                if i==0:
                    protocolName = os.path.dirname(self.kwdFileList[j])
                    ax[-1].set_title(protocolName[protocolName.rfind('_')+1:],fontsize='x-small')
                if i==len(units)-1 and j==len(protocol)-1:
                    ax[-1].set_xticks([0,maxInterval])
                    ax[-1].set_xlabel('ISI (s)',fontsize='x-small')
                else:
                    ax[-1].set_xticks([])
            for j,a in enumerate(ax):
                a.set_ylim([0,ymax])
                if j==0:
                    a.set_yticks([int(ymax*100)/100])
                    a.set_ylabel(u,fontsize='x-small')
                else:
                    a.set_yticks([])
                    
    
    def plotSDF(self,unit,protocol,startSamples=None,offset=0,windowDur=None,sigma=0.02,sampInt=0.001,paramNames=None):
        # offset in seconds
        # windowDur input in seconds then converted to samples
        if paramNames is not None and len(paramNames)>2:
            raise ValueError('plotSDF does not accept more than 2 parameters')
        protocol = str(protocol)
        if windowDur is not None:
            windowDur = int(windowDur*self.sampleRate)
        params = []
        if startSamples is None:
            p = self.visstimData[str(protocol)]
            try:
                trialStartFrame = p['trialStartFrame']
            except:
                trialStartFrame = p['stimStartFrames']
            startSamples = p['frameSamples'][trialStartFrame]
            if windowDur is None:
                windowDur = np.diff(startSamples)
            startSamples = startSamples[:-1]
            if paramNames is not None:
                for name in paramNames:
                    params.append(p[name][:-1])
        else:
            windowDur = [windowDur for _ in startSamples]
        paramSet = [np.unique(param) for param in params]
        startSamples += int(offset*self.sampleRate)
        spikes = self.units[str(unit)]['times'][protocol]
        
        if len(params)==0:
            rows = cols = [0]
        elif len(params)==1:
            rows = range(len(set(params[0])))
            cols = [0]
        else:
            cols,rows = [range(len(set(params[i]))) for i in (0,1)]
        plt.figure(facecolor='w')
        gs = gridspec.GridSpec(len(rows),len(cols))
        ax = []
        xmax = max(windowDur)/self.sampleRate+offset
        ymax = 0
        for i in rows:
            if len(params)>0:
                trials = np.where(params[len(params)-1]==paramSet[len(params)-1][::-1][i])[0]
            else:
                trials = np.arange(len(startSamples))
            for j in cols:
                if len(params)>1:
                    trialIndex = np.intersect1d(trials,np.where(params[0]==paramSet[0][j])[0])
                else:
                    trialIndex = trials
                sdf,t = self.getMeanSDF(spikes,startSamples[trialIndex],max(windowDur[trialIndex]),sigma=sigma)
                ymax = max(ymax,sdf.max())
                ax.append(plt.subplot(gs[i,j]))
                ax[-1].plot(t+offset,sdf)
        ymax *= 1.05
        for ind,a in enumerate(ax):
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            a.set_xlim([offset,xmax])
            a.set_ylim([0,ymax])
            if ind==len(ax)-1:
                a.set_xticks([0,xmax])
                a.set_xlabel('Time (s)',fontsize='x-small')
            else:
                a.set_xticks([])
            if ind==0:
                a.set_yticks([0,ymax])
                a.set_ylabel('Spikes/s',fontsize='x-small')
                a.set_title('Unit '+str(unit),fontsize='small')
            else:
                a.set_yticks([])
                
                
    def getMeanSDF(self,spikes,startSamples,windowSamples,sigma=0.02,sampInt=0.001):
        binSamples = sampInt*self.sampleRate
        bins = np.arange(0,windowSamples+binSamples,binSamples)
        binnedSpikeCount = np.zeros((len(startSamples),len(bins)-1))
        for i,start in enumerate(startSamples):
            binnedSpikeCount[i],_ = np.histogram(spikes[np.logical_and(spikes>=start,spikes<=start+windowSamples)]-start,bins)
        sdf = np.mean(scipy.ndimage.filters.gaussian_filter1d(binnedSpikeCount,sigma/sampInt,axis=1),axis=0)/sampInt
        t = bins[:-1]/self.sampleRate
        return sdf,t
            
    
    def plotRaster(self,unit,protocol,startSamples=None,offset=0,windowDur=None,paramNames=None,paramColors=None,grid=False):
        # offset and windowDur input in seconds then converted to samples
        protocol = str(protocol)
        offset = int(offset*self.sampleRate)
        if windowDur is not None:
            windowDur = int(windowDur*self.sampleRate)
        params = []
        if startSamples is None:
            p = self.visstimData[str(protocol)]
            try:
                trialStartFrame = p['trialStartFrame']
            except:
                trialStartFrame = p['stimStartFrames']
            startSamples = p['frameSamples'][trialStartFrame]
            if windowDur is None:
                windowDur = np.diff(startSamples)
            startSamples = startSamples[:-1]
            if paramNames is not None:
                for name in paramNames:
                    params.append(p[name][:-1])
        else:
            windowDur = [windowDur for _ in startSamples]
        startSamples += offset
        spikes = self.units[str(unit)]['times'][protocol]
        
        if paramColors is None:
            paramColors = [None]*len(params)
        else:
            for i,c in enumerate(paramColors):
                if c=='auto' and i<len(params):
                    paramColors[i] = cm.Dark2(range(0,256,int(256/len(set(params[i])))))
                    break
        grid = True if grid and len(paramNames)==2 else False
         
        plt.figure(facecolor='w')
        if grid:
            axes = []
            rows = []
            gs = gridspec.GridSpec(len(set(params[1])),len(set(params[0])))
        else:
            axes = [plt.subplot(1,1,1)]
            rows = [0]
            gs = None
        if len(params)<1:
            self.appendToRaster(axes,spikes,startSamples,offset,windowDur,rows=rows)
        else:
            self.parseRaster(axes,spikes,startSamples,offset,windowDur,params,paramColors,rows=rows,grid=grid,gs=gs)
        for ax,r in zip(axes,rows):
            ax.set_xlim([offset/self.sampleRate,(max(windowDur)+offset)/self.sampleRate])
            ax.set_ylim([-0.5,r+0.5])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            if grid:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Trial')
        axes[-1].set_title('Unit '+str(unit))
         
         
    def parseRaster(self,axes,spikes,startSamples,offset,windowDur,params,paramColors,paramIndex=0,trialsIn=None,rows=[0],grid=False,gs=None,grow=None,gcol=0):
        paramVals = np.unique(params[paramIndex])
        if grid and grow is None:
            grow = gs.get_geometry()[0]-1
        for i,val in enumerate(paramVals):
            if grid:
                axes.append(plt.subplot(gs[grow,gcol]))
                rows.append(0)
                grow -= 1
            trialIndex = np.where(params[paramIndex]==val)[0]
            if trialsIn is not None:
                trialIndex = np.intersect1d(trialsIn,trialIndex)
            if paramIndex<len(params)-1:
                if paramColors[paramIndex] is not None:
                    paramColors[paramIndex+1] = [paramColors[paramIndex][i]]*len(set(params[paramIndex+1]))
                self.parseRaster(axes,spikes,startSamples,offset,windowDur,params,paramColors,paramIndex+1,trialIndex,rows,grid,gs,None,gcol)
                if grid:
                    gcol += 1
            else:
                color = 'k' if paramColors[paramIndex] is None else paramColors[paramIndex][i]
                self.appendToRaster(axes,spikes,startSamples,offset,windowDur,trialIndex,rows,color)
 
 
    def appendToRaster(self,axes,spikes,startSamples,offset,windowDur,trialIndex=None,rows=[0],color='k'):
        if trialIndex is None:
            trialIndex = range(len(startSamples))
        for i in trialIndex:
            spikeTimes = (spikes[np.logical_and(spikes>startSamples[i],spikes<startSamples[i]+windowDur[i])]-startSamples[i]+offset)/self.sampleRate
            axes[-1].vlines(spikeTimes,rows[-1]-0.4,rows[-1]+0.4,color=color)
            rows[-1] += 1
    
    
    def runAllAnalyses(self, units=None, protocolsToRun = ['sparseNoise', 'gratings', 'gratings_ori', 'spots', 'checkerboard'], splitRunning = False, useCache=False):

        for pro in protocolsToRun:
            protocol = self.getProtocolIndex(pro)
            
            if 'gratings'==pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['stimTime']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=statTrials, saveTag='_stat')
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=runTrials, saveTag='_run')
                else:
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf')

            elif 'gratings_ori'==pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['stimTime']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=statTrials, saveTag='_stat')
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=runTrials, saveTag='_run')
                else:
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori')

            elif 'sparseNoise' in pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['boxDuration']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.findRF(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag='_stat')
                    self.findRF(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag='_run')
                else:                    
                    self.findRF(units, protocol=protocol, useCache=useCache)
            elif 'spots' in pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['trialStartFrame'][:-1]
                    trialDuration = (self.visstimData[str(protocol)]['trialNumFrames']).astype(np.int)
                    trialEndFrames = trialStartFrames + trialDuration[:-1]
                    frameSamples = self.visstimData[str(protocol)]['frameSamples']     
                    trialStarts = frameSamples[trialStartFrames]
                    trialEnds = frameSamples[trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag='_stat')
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag='_run')
                else:
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache)
            elif 'checkerboard' in pro:
                if splitRunning:
                    trialStartFrame = self.visstimData[str(protocol)]['trialStartFrame'][:-1]
                    trialDuration = (self.visstimData[str(protocol)]['trialNumFrames']).astype(int)
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame+trialDuration]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeCheckerboard(units, protocol=protocol, trials=statTrials, saveTag='_stat')
                    self.analyzeCheckerboard(units, protocol=protocol, trials=runTrials, saveTag='_run')
                else:
                    self.analyzeCheckerboard(units, protocol=protocol)
            else:
                print("Couldn't find analysis script for protocol type:", pro)
                
     
    def getProtocolIndex(self, label):
        protocol = []
        protocol.extend([i for i,f in enumerate(self.kwdFileList) if ntpath.dirname(f).endswith(label)])
        if len(protocol)<1:
            raise ValueError('No protocols found matching: '+label)
        elif len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+label)
        return protocol[0]
        
          
    def getOrderedUnits(self,units=None):
        # orderedUnits, yPosition = self.getOrderedUnits(units)
        if units is None:
            units = [u for u in self.units]
        elif isinstance(units,list):
            units = [str(u) for u in units]
        else:
            units = [str(units)]
        for u in units:
            if u not in self.units:
                units.remove(u)
                print(str(u)+' not in units')
        if len(units)<1:
            raise ValueError('Found no matching units')
        orderedUnits = [(u,self.units[u]['ypos']) for u in self.units if u in units]
        orderedUnits.sort(key=lambda i: i[1], reverse=True)
        return zip(*orderedUnits)
    
    
    def getSingleUnits(self, fileDir = None, protocolsToAnalyze = None):
        if fileDir is None:
            fileDir = getDir()
        fileList, nsamps = getKwdInfo(dirPath=fileDir)
        if protocolsToAnalyze is None:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir)
        else:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir, protocolsToAnalyze=protocolsToAnalyze)
    
    
    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
        from load_phy_template import load_phy_template
                 
        if fileDir is None:
            fileDir = getDir()
        
        if protocolsToAnalyze is None:
            protocolsToAnalyze = np.arange(len(self.d))
        
        self.units = load_phy_template(fileDir, sampling_rate = self.sampleRate)
        for unit in self.units:
            spikeTimes = (self.units[unit]['times']).astype(int)
           
            if kwdNsamplesList is not None:
                self.units[unit]['times'] = {}
                protocolEnds = np.cumsum(kwdNsamplesList)
                protocolStarts = np.insert(protocolEnds, 0, 0)[:-1] - 1
                for pro in protocolsToAnalyze:                    
                    self.units[unit]['times'][str(pro)] = spikeTimes[np.logical_and(spikeTimes >= protocolStarts[pro], spikeTimes < protocolEnds[pro])]
                    self.units[unit]['times'][str(pro)] -= protocolStarts[pro]
            else:
              self.units[unit]['times'] = spikeTimes       


    def saveHDF5(self, fileSaveName = None, fileOut = None, saveDict = None, grp = None):
        if fileSaveName is None and fileOut is None:
            fileSaveName = saveFile()
            if fileSaveName=='':
                return
            fileOut = h5py.File(fileSaveName, 'w')
        elif fileSaveName is not None and fileOut is None:            
            fileOut = h5py.File(fileSaveName,'w')

        if saveDict is None:
            saveDict = self.__dict__
        if grp is None:    
            grp = fileOut['/']
        
        for key in saveDict:    
            if type(saveDict[key]) is dict:
                self.saveHDF5(fileOut=fileOut, saveDict=saveDict[key], grp=grp.create_group(key))
            else:
                try:
                    grp[key] = saveDict[key]
                except:
                    try:
                        grp.create_dataset(key,data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
                    except:
                        print('Could not save: ', key)
                    
                    
    def loadHDF5(self, fileName=None, grp=None, loadDict=None):
        if fileName is None and grp is None:        
            fileName = getFile()
            if fileName=='':
                return
        if grp is None:
            grp = h5py.File(str(fileName))
        for key,val in grp.items():
            if isinstance(val,h5py._hl.dataset.Dataset):
                v = val.value
                if isinstance(v,np.ndarray) and v.dtype==np.object:
                    v = v.astype('U')
                if loadDict is None:
                    setattr(self,key,v)
                else:
                    loadDict[key] = v
            elif isinstance(val,h5py._hl.group.Group):
                if loadDict is None:
                    setattr(self,key,{})
                    self.loadHDF5(grp=val,loadDict=getattr(self,key))
                else:
                    loadDict[key] = {}
                    self.loadHDF5(grp=val,loadDict=loadDict[key])
                    
    
    def saveWorkspace(self, variables=None, saveGlobals = False, fileName=None, exceptVars = []):
        if fileName is None:
            fileName = saveFile()
            if fileName=='':
                return
        shelf = shelve.open(fileName, 'n')
        
        if variables is None:
            if not saveGlobals:
                variables = self.__dict__.keys()
            else:
                variables = self.__dict__.keys() + globals().keys()
        
        for key in variables:
            try:
                if key in self.__dict__.keys():
                    shelf[key] = self.__dict__[key]
                else:
                    shelf[key] = globals()[key]    
            except TypeError:
                # __builtins__, my_shelf, and imported modules can not be shelved.
                print('ERROR shelving: {0}'.format(key))
        shelf.close()


    def loadWorkspace(self, fileName = None):
        if fileName is None:        
            fileName = getFile()
            if fileName=='':
                return
        shelf = shelve.open(fileName)
        for key in shelf:
            setattr(self, key, shelf[key])
        shelf.close()


    def readExcelFile(self, sheetname, fileName = None):
        if fileName is None:        
            fileName = getFile()
            if fileName=='':
                return        
        
        table = pandas.read_excel(fileName, sheetname=sheetname, parse_cols="A, B")
        
        for u in xrange(table.shape[0]):
            unit = table.Cell[u]
            label = table.Label[u]
            self.units[str(unit)]['label'] = label


    def getUnitsByLabel(self, labelID, criterion, notFlag=False):
        units = []
        for unit in self.units.keys():
            label = self.units[unit][labelID]
            if notFlag:                
                if label not in criterion:
                    units.append(unit)
            else:
                if label in criterion:
                    units.append(unit)
        return units
        
        
    def findCCFCoords(self, tipPos, entryPos, tipProbePos=-1300):
        xRange = entryPos[0] - tipPos[0]
        yRange = entryPos[1] - tipPos[1]
        zRange = entryPos[2] - tipPos[2]
        
        trackLength = math.sqrt(xRange**2 + yRange**2 + zRange**2)
        
        xSlope = xRange/trackLength
        ySlope = yRange/trackLength
        zSlope = zRange/trackLength
                        
        units, unitsYPos = self.getOrderedUnits(self.units.keys())
        for i, unit in enumerate(units):
            distFromTip = unitsYPos[i] - tipProbePos
            pos = np.array([xSlope, ySlope, zSlope])*distFromTip
            pos+=tipPos
            
            self.units[str(unit)]['CCFCoords'] = pos
            
        
# utility functions

def getFile():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getOpenFileName(None,'Choose File')
    
    
def getDir(rootDir=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getExistingDirectory(None,'Choose Directory',rootDir) 
    

def saveFile():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getSaveFileName(None,'Save As')


def getKwdInfo(dirPath=None):
    # kwdFiles, nSamples = getKwdInfo()
    # returns kwd file paths and number of samples in each file ordered by file start time
    if dirPath is None:    
        dirPath = getDir(dataDir)
        if dirPath == '':
            return
    kwdFiles = []
    startTime = []
    nSamples = []
    for item in os.listdir(dirPath):
        itemPath = os.path.join(dirPath,item)
        if os.path.isdir(itemPath):
            for f in os.listdir(itemPath):
                if f[-4:]=='.kwd':
                    startTime.append(datetime.datetime.strptime(os.path.basename(itemPath)[0:19],'%Y-%m-%d_%H-%M-%S'))
                    kwdFiles.append(os.path.join(itemPath,f))
                    kwd = h5py.File(kwdFiles[-1],'r')
                    nSamples.append(kwd['recordings']['0']['data'].shape[0])
    return zip(*[n[1:] for n in sorted(zip(startTime,kwdFiles,nSamples),key=lambda z: z[0])])


def makeDat(kwdFiles=None):
    if kwdFiles is None:
        kwdFiles, _ = getKwdInfo()
    dirPath = os.path.dirname(os.path.dirname(kwdFiles[0]))
    datFilePath = os.path.join(dirPath,os.path.basename(dirPath)+'.dat')
    datFile = open(datFilePath,'wb')
    for filenum, filePath in enumerate(kwdFiles):
        print('Copying kwd file ' + str(filenum + 1) + ' of ' + str(len(kwdFiles)) + ' to dat file')
        kwd = h5py.File(filePath,'r')
        dset = kwd['recordings']['0']['data']
        i = 0
        while i<dset.shape[0]:
            (dset[i:i+dset.chunks[0],:128]).tofile(datFile)                        
            i += dset.chunks[0]
    datFile.close()
    copyPath = r'\\10.128.38.3\data_local_1\corbett'
    print('Copying dat file to ' + copyPath)
    shutil.copy(datFilePath,copyPath)
    
    
def gauss2D(xyTuple,x0,y0,sigX,sigY,theta,amplitude):
    x,y = xyTuple # (x,y)
    y = y[:,None]                                                                                                             
    a = (math.cos(theta)**2)/(2*sigX**2)+(math.sin(theta)**2)/(2*sigY**2)   
    b = (math.sin(2*theta))/(4*sigX**2)-(math.sin(2*theta))/(4*sigY**2)    
    c = (math.sin(theta)**2)/(2*sigX**2)+(math.cos(theta)**2)/(2*sigY**2)   
    z = amplitude * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))                                   
    return z.ravel()


def fitGauss2D(x,y,data,initialParams):
    try:
        gridSize = max(x[-1]-x[0],y[-1]-y[0])
        maxOffGrid = 0.2*gridSize
        lowerBounds = np.array([x[0]-maxOffGrid,y[0]-maxOffGrid,0,0,0,0])
        upperBounds = np.array([x[-1]+maxOffGrid,y[-1]+maxOffGrid,gridSize,gridSize,2*math.pi,1.5*data.max()])
        fitParams,fitCov = scipy.optimize.curve_fit(gauss2D,(x,y),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except RuntimeError:
        print('fit failed')
        return
    # fitData = gauss2D((x,y),*fitParams).reshape(y.size,x.size)
    return fitParams


def getEllipseXY(x,y,a,b,angle):
    sinx = np.sin(np.arange(0,361)*math.pi/180)
    cosx = np.cos(np.arange(0,361)*math.pi/180)
    X = x+a*cosx*math.cos(angle)-b*sinx*math.sin(angle)
    Y = y+a*cosx*math.sin(angle)+b*sinx*math.cos(angle)
    return X,Y
    

def stfLogGauss2D(stfTuple,sf0,tf0,sigSF,sigTF,speedTuningIndex,amplitude):
    sf,tf = stfTuple
    tf = tf[:,None]
    z = amplitude * np.exp(-((np.log2(sf)-np.log2(sf0))**2)/(2*sigSF**2)) * np.exp(-((np.log2(tf)-(speedTuningIndex*(np.log2(sf)-np.log2(sf0))+np.log2(tf0)))**2)/(2*sigTF**2))
    return z.ravel()


def fitStfLogGauss2D(sf,tf,data,initialParams):
    try:
        lowerBounds = np.array([0,0,0,0,-0.5,0])
        upperBounds = np.array([1,16,np.inf,np.inf,1.5,1.5*initialParams[-1]])
        fitParams,fitCov = scipy.optimize.curve_fit(stfLogGauss2D,(sf,tf),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except RuntimeError:
        print('fit failed')
        return
    # fitData = stfLogGauss2D((sf,tf),*fitParams).reshape(tf.size,sf.size)
    return fitParams


def getStfContour(sf,tf,fitParams):
    intpPts = 100
    sfIntp = np.logspace(np.log2(sf[0]*0.5),np.log2(sf[-1]*2),intpPts,base=2)
    tfIntp = np.logspace(np.log2(tf[0]*0.5),np.log2(tf[-1]*2),intpPts,base=2)
    intpFit = stfLogGauss2D((sfIntp,tfIntp),*fitParams).reshape(intpPts,intpPts)
    thresh = 0.6065*intpFit.max() # one stdev
    contourLine = np.full((2*intpPts,2),np.nan)
    for i in range(len(sfIntp)):
        c = np.where(intpFit[:,i]>thresh)[0]
        if len(c)>0:
            contourLine[i,0] = sfIntp[i]
            contourLine[i,1] = tfIntp[c[0]]
            contourLine[-1-i,0] = sfIntp[i]
            contourLine[-1-i,1] = tfIntp[c[-1]]
    contourLine = contourLine[np.logical_not(np.isnan(contourLine[:,0])),:]
    if contourLine.shape[0]>0:
        contourLine = np.concatenate((contourLine,contourLine[0,:][None,:]),axis=0)
    x,y = (np.log2(contourLine)-np.log2([sfIntp[0],tfIntp[0]])).T-1
    return x,y
    
    
def getDSI(resp,theta):
    theta = np.copy(theta)*math.pi/180
    sumX = np.sum(resp*np.cos(theta))
    sumY = np.sum(resp*np.sin(theta))
    dsi = np.sqrt(sumX**2+sumY**2)/resp.sum()
    prefTheta = ((math.atan2(sumY,sumX)*180/math.pi)+360)%360
#    prefTheta = math.atan(sumY/sumX)*180/math.pi
#    if sumX<0:
#        if sumY>0:
#            prefTheta += 180
#        else:
#            prefTheta += 180
#    elif sumY<0:
#        prefTheta += 360
    return dsi, prefTheta


def findFWatHM(data):
    data = np.array(data)    
    data -= data.min()
    maxPoint = np.argmax(data)
    maxVal = data[maxPoint]
    hMax = maxVal/2.0
    
    hStart = np.nan
    for i, point in enumerate(data[maxPoint::-1]):
        if point <= hMax:
            hStart = maxPoint - i
            break
    
    hEnd = np.nan  
    for i, point in enumerate(data[maxPoint:]):
        if point <= hMax:
            hEnd = i + maxPoint
            break
        
    return [hStart, hEnd], hEnd-hStart



if __name__=="__main__":
    pass       