# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:01:44 2019

@author: chelsea.strawder

Prints out all the information in the daily summary, including:
- certain parameters from the behavioral training that day
- counts of repeat, ignored, and usable trials 
-  fraction of trials that are correct, incorrect, and no response for each side 
(left stim==turn right, right stim==turn left)
- percent correct and rewards earned 


"""


import numpy as np
from dataAnalysis import create_df


def session_stats(d, nogo=False, returnAs:'str_array'):    #returnAs 'str_array', 'num_array', or 'print'

    print(str(d) + '\n')
    
    df = create_df(d)
    
    sessionDuration = d['trialResponseFrame'][-1]   # in frames
    
    catchTrials = df['catchTrial'].value_counts()[1]
    notCatch = df['catchTrial'].value_counts()[0]
    
    assert (len(df) - catchTrials) == (notCatch), "catch trial issue"
        
    totalTrials = notCatch
    
    rightTotal = df[(df['rewDir']==1) & (df['repeat']==False) & (df['catchTrial']==False)]  #ignored included
    right = rightTotal[rightTotal['ignoreTrial']==False]
    
    rightCorrect = len(right[right['resp']==1])
    rightIncorrect = len(right[right['resp']==-1])
    rightNoResp = len(right[right['resp']==0])
    
    
    leftTotal = df[(df['rewDir']==-1) & (df['repeat']==False) & (df['catchTrial']==False)]  #ignored included
    left = leftTotal[leftTotal['ignoreTrial']==False]
    
    leftCorrect = len(left[left['resp']==1])
    leftIncorrect = len(left[left['resp']==-1])
    leftNoResp = len(left[left['resp']==0])
    
    ignored = df.groupby(['rewDir'])['ignoreTrial'].sum(sorted=True) # num of ignore trials by side and total
    repeats = df[(df['repeat']==True) & (df['catchTrial']==False) & (df['ignoreTrial']==False)]
    
    usableTrialTotal = (len(left) + len(right))   # no ignores, repeats, or catch
    respTotal = usableTrialTotal - (rightNoResp + leftNoResp)
    totalCorrect = (rightCorrect + leftCorrect)

    if returnAs == 'str_array'.lower():
        
            array = ['Wheel Reward Dist: ' + str(d['wheelRewardDistance'][()]),
                     'Norm reward: ' + str(d['normRewardDistance'][()]),
                     'Max wait frames: ' + str(d['maxResponseWaitFrames'][()]),
                     'Prob go right: ' + str(d['probGoRight'][()]),
                     'Session duration (mins): ' + str(np.round(sessionDuration/df.framerate/60, 2)),
                     '/n/',
                     'Repeats: ' + str(len(repeats)) + '/' + str(totalTrials),
                     'Total Ignore: ' + str(ignored[1]+ignored[-1]),
                     'Performance trials: ' + str(usableTrialTotal),
                     'Right Ignore: ' + str(ignored[1]),
                     'Right trials: ' + str(len(right)),
                     'R % Correct: ' + str(np.round(rightCorrect/len(right), 2)),
                     'R % Incorrect: ' + str(np.round(rightIncorrect/len(right),2)),
                     'R % No Resp: ' + str(np.round(rightNoResp/len(right),2)),
                     'Left Ignore: ' + str(ignored[-1]),
                     'Left trials: ' + str(len(left)),
                     'L % Correct: ' + str(np.round(leftCorrect/len(left), 2)),
                     'L % Incorrect: ' + str(np.round(leftIncorrect/len(left),2)),
                     'L % No Resp: ' + str(np.round(leftNoResp/len(left),2)),
                     'Total Correct, given Response: ' + str(np.round((leftCorrect+rightCorrect)/respTotal,2)),
                     'Total Correct: ' + str(np.round(totalCorrect/usableTrialTotal,2)),
                     'Rewards this session:  ' + str(len(df[df['resp']==1])),
                     '\n']
            
            return array 
        
        
    elif returnAs == 'num_array'.lower():   # returns array of ints and floats, no titles or words
        
         array = [d['wheelRewardDistance'][()],
                  d['normRewardDistance'][()],
                  d['maxResponseWaitFrames'][()],
                  d['probGoRight'][()],
                  np.round(sessionDuration/df.framerate/60, 2),
                  len(repeats)/(totalTrials),
                  ignored[1]+ignored[-1],
                  usableTrialTotal,
                  ignored[1],
                  len(right),
                  np.round(rightCorrect/len(right), 2),
                  np.round(rightIncorrect/len(right),2),
                  np.round(rightNoResp/len(right),2),
                  ignored[-1],
                  len(left),
                  np.round(leftCorrect/len(left), 2),
                  np.round(leftIncorrect/len(left),2),
                  np.round(leftNoResp/len(left),2),
                  np.round((leftCorrect+rightCorrect)/respTotal,2),
                  np.round(totalCorrect/usableTrialTotal,2),
                  len(df[df['resp']==1])]
         
         return array
         
    else:

        if 'wheelRewardDistance' in d.keys():
            print('Wheel Reward Dist: ' + str(d['wheelRewardDistance'][()]))
        
        print('Norm reward: ' + str(d['normRewardDistance'][()]))
        print('Max wait frames: ' + str(d['maxResponseWaitFrames'][()]))
        print('Prob go right: ' + str(d['probGoRight'][()]))
        print('Session duration (mins): ' + str(np.round(sessionDuration/df.framerate/60, 2)))
        print('\n')
        print('Repeats: ' + str(len(repeats)) + '/' + str(totalTrials))
        print('Total Ignore: ' + str(ignored[1]+ignored[-1]))
        print('Performance trials: ' + str(usableTrialTotal))
        print('\n')
        print("Right Ignore: " + str(ignored[1]))
        print("Right trials: " + str(len(right)))
        print("R % Correct: " + str(np.round(rightCorrect/len(right), 2)))
        print("R % Incorrect: " + str(np.round(rightIncorrect/len(right),2)))
        print("R % No Resp: " + str(np.round(rightNoResp/len(right),2)))
        print('\n') 
        print("Left Ignore: " + str(ignored[-1]))
        print("Left trials: " + str(len(left)))
        print("L % Correct: " + str(np.round(leftCorrect/len(left), 2)))
        print("L % Incorrect: " + str(np.round(leftIncorrect/len(left),2)))
        print("L % No Resp: " + str(np.round(leftNoResp/len(left),2)))
        print('\n')
        print('Total Correct, given Response: ' + str(np.round((leftCorrect+rightCorrect)/respTotal,2)))
        print('Total Correct: ' + str(np.round(totalCorrect/usableTrialTotal,2)))
        print("Rewards this session:  " + str(len(df[df['resp']==1])))  # all rewards, including repeats and ignored 
        # can use to calculate how much water mouse recieved in session 
        print('\n')
        
    
    
    
    
#######  make this a function? 
#    
#    if nogo==True :
#        no_goTotal = len(trialTargetFrames[trialTargetFrames==0])
#        no_goCorrect = len(trialResponse[(trialResponse==1) & (trialTargetFrames==0)]) 
#        print('No-go Correct:  ' + str(round(no_goCorrect/no_goTotal, 2)*100) + '% of ' + str(no_goTotal))
#        
#    #returns an array of values that show the direction turned for ALL no-go trials, then returns % per direction  
#        no_goTurnDir = []
#    
#        stimStart = d['trialStimStartFrame'][:-1] 
#                                                        # this accounts for those trials where the trial started then the session ended
#        if len(stimStart)==len(prevTrialIncorrect):     # otherwise the arrays are different lengths and can't be indexed
#            pass
#        else:
#            stimStart= d['trialStimStartFrame'][:]
#            trialRespFrames = d['trialResponseFrame'][:]
#            trialOpenLoop = d['trialOpenLoopFrames'][:len(stimStart)] 
#            deltaWheel = d['deltaWheelPos'][:]
#    
#        if ignore.upper()== 'YES': 
#           stimStart = stimStart[prevTrialIncorrect==False]
#           trialRespFrames = trialRespFrames[prevTrialIncorrect==False]
#           trialOpenLoop = trialOpenLoop[prevTrialIncorrect==False]
#    
#        stimStart = stimStart[trialTargetFrames==0]
#        trialRespFrames = trialRespFrames[trialTargetFrames==0]
#        trialOpenLoop = trialOpenLoop[trialTargetFrames==0]
#        deltaWheel = d['deltaWheelPos'][:]
#        no_goResp = trialResponse[trialTargetFrames==0]
#        
#        stimStart += trialOpenLoop
#        
#        startWheelPos = []
#        endWheelPos = []
#        
#        for (start, end, resp) in zip(stimStart, trialRespFrames, no_goResp):
#            if resp==-1:
#                endWheelPos.append(deltaWheel[end])
#                startWheelPos.append(deltaWheel[start])
#            
#        endWheelPos = np.array(endWheelPos)
#        startWheelPos = np.array(startWheelPos)   
#        wheelPos = endWheelPos - startWheelPos
#        
#        for i in wheelPos:
#            if i >0:
#                no_goTurnDir.append(1)
#            else:
#                no_goTurnDir.append(-1)
#        
#        no_goTurnDir = np.array(no_goTurnDir)
#        print('no-go turn R:  ' + str(sum(no_goTurnDir==1)))
#        print('no-go turn L:  ' + str(sum(no_goTurnDir==-1)))
#    else:
#        print('*There were no nogos')
#        
#########        

    
    

    
    
    
