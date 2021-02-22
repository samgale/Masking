# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:50:03 2019

@author: svc_ccg
"""

import MaskingTask
import TaskControl

task = MaskingTask.MaskingTask('pilot')
gray = TaskControl.TaskControl('pilot')
gray.saveParams = False

## all of these frames are for a 120 Hz monitor

   #for position task (stim on either side of screen)

task.setDefaultParams('training2', 'pos')
task.normRewardDistance=.2
task.openLoopFramesFixed=24
task.nogoWaitFrames=84
task.maxResponseWaitFrames=60
task.probNoGo=.2
task.incorrectTrialRepeats = 3
task.maxConsecutiveSameDir = task.incorrectTrialRepeats + 3


## training5 transition 
task.setDefaultParams('training4', 'pos')
task.maxResponseWaitFrames = 120
task.wheelRewardDistance = 4.0


task.setDefaultParams('training5', 'pos')

task.targetContrast = [.4,.8]
task.targetFrames = [2, 4,6]

task.setDefaultParams('training3', 'pos')
task.wheelRewardDistance = 4.0  #(to prevent stim jitter)
task.maxResponseWaitFrames = 300
task.start('563901')

    # rotation task

task.setDefaultParams('training4', 'rot')   # for orientation rotation task
task.useGoTone=False
task.maxResponseWaitFrames = 120
task.quiescentFrames = 60
task.autoRotationRate = 0
task.incorrectTimeoutFrames = 720 
task.incorrectTrialRepeats = 3
task.gratingRotationGain = .08
gray.start()


    #for rotation transition to masking  

task.moveStim=False
task.postRewardTargetFrames=0
task.targetFrames=[6]
task.targetContrast = [.1,.2,.3,.4,.6,1]
task.probGoRight=.5
task.rewardRotation=30 
task.incorrectTimeoutFrames = 0
task.incorrectTrialRepeats = 0
task.useIncorrectNoise = False
   

    # for position transitioning to masking

task.setDefaultParams('training5', 'pos') 
task.moveStim = False
task.postRewardTargetFrames = 0
task.targetFrames = [2]
task.normRewardDistance = .12  
task.useIncorrectNoise = False
task.incorrectTimeoutFrames = 0
task.incorrectTrialRepeats = 0 
task.preStimFramesFixed = 240
task.preStimFramesMax = 480  
task.targetContrast = [.2,.4,.6,.8,1]




    #add masking

task.setDefaultParams('training5', 'pos')
task.normRewardDistance = .12                 #position task
task.probNoGo = .2        # dont use for rotation mice
task.nogoWaitFrames = 60  

task.setDefaultParams('training4', 'rot') 
task.rewardRotation = 30      
task.gratingRotationGain = .08               #rotation task


task.moveStim=False 
task.postRewardTargetFrames=0
task.maskType = 'plaid'
task.maskShape = 'target'
task.maskFrames = [24]
task.maskContrast = [1]
task.probMask = .6
task.targetFrames = [2]
task.targetContrast = [.5]
task.maskOnset = [2,3,4,6,12]
task.useIncorrectNoise=False
task.incorrectTimeoutFrames = 0
task.incorrectTrialRepeats=0
 








##############################################################################









 

# for testing target duration
task.setDefaultParams('training5')
task.moveStim = False
task.useIncorrectNoise = False
task.incorrectTimeoutFrames = 0
task.incorrectTrialRepeats = 0 
task.postRewardTargetFrames = 0
task.normRewardDistance = .16     # adjust this each mouse
task.targetFrames = [1,2,4,8,16]



#set varying probabilites for biases 
task.fracTrialsGoRight = .33
task.fracTrialsNoGo = 0


  
# to create target drift
task.reverseTargetPhase = True
task.reversePhasePeriod = 25
 
task.quiescentFrames = [84, 200]
task.maxResponseWaitFrames = 64
task.incorrectTimeoutFrames = 0

task.incorrectTrialRepeats = 0

task.targetFrames = [4]
task.moveStim = False
task.maskType = None
task.postRewardTargetFrames = 0

#for masking 
task.maskType = 'plaid'
task.maskContrast = [.025, 0.05, .1, .2, .4]
task.maskFrames = 8
task.maskOnset = [0, 4, 10, 24] 

task.preStimFrames = [360,480]

