
mo = 2
rasterIpsi = []
rasterContra = []
countIpsi = [0]
countContra = [0]
for obj in exps:
    trials = ~obj.longFrameTrials & (obj.trialType=='mask') & (obj.maskOnset==mo)
    rewardDir = (-1,1) if hasattr(obj,'hemi') and obj.hemi=='right' else (1,-1)
    for raster,count,rd in zip((rasterIpsi,rasterContra),(countIpsi,countContra),rewardDir):
        startTimes = obj.frameSamples[obj.stimStart[trials & (obj.rewardDir==rd)]+obj.frameDisplayLag]/obj.sampleRate-preTime
        for u in obj.goodUnits:
            if respUnits[count[0]]:
               spikeTimes = obj.units[u]['samples']/obj.sampleRate
               t = np.random.choice(startTimes)
               raster.append(spikeTimes[(spikeTimes>t) & (spikeTimes<t+windowDur)]-t-preTime)
            count[0] += 1
    

for raster,clr in zip((rasterIpsi,rasterContra),'br'):
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1,1,1)
    rasterShuffled = raster.copy()
    random.shuffle(rasterShuffled)
    for i,r in enumerate(rasterShuffled):
        ax.vlines(r,i-0.5,i+0.5,clr)
    for side in ('right','top','left','bottom'):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-0.1,0.6])
    plt.tight_layout()

