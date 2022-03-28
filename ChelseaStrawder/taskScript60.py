# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:01:30 2020

@author: svc_ccg

Box5 monitor has 60hz framerate**
"""

import MaskingTask
import TaskControl

gray = TaskControl.TaskControl('box5')
task = MaskingTask.MaskingTask('box5')
gray.saveParams = False

task.setDefaultParams('training3', 'pos')
task.rewardSizeLeft = None
task.rewardSizeRight= None

## don't forget to run these for 60 frame monitor!!
task.preStimFramesFixed = task.preStimFramesFixed/2
task.preStimFramesVariableMean = task.preStimFramesVariableMean/2
task.preStimFramesMax = task.preStimFramesMax/2
task.quiescentFrames = task.quiescentFrames/2
task.openLoopFramesFixed = task.openLoopFramesFixed/2
task.openLoopFramesVariableMean = task.openLoopFramesVariableMean/2
task.openLoopFramesMax = task.openLoopFramesMax/2
task.incorrectTimeoutFrames = task.incorrectTimeoutFrames/2

task.wheelRewardDistance=  6.0
task.incorrectTrialRepeats =5
task.maxResponseWaitFrames = 600


#  new mice 
task.maxResponseWaitFrames = 300

task.incorrectTimeoutFrames = 180
task.solenoidOpenTime = .08


task.maxResponseWaitFrames = 60
