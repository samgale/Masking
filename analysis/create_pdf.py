# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:54:31 2020

@author: chelsea.strawder
"""

import performanceBySOA
import dataAnalysis
from qualityControl import check_qviolations
from percentCorrect import session_stats
from catchTrials import catch_trials
import matplotlib.pyplot as plt
import os
import reportlab
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from datetime import datetime


def create_daily_summary(d):


    mouse_id=d['subjectName'][()]
    date = d['startTime'][()].split('_')[0][-4:]
    date = date[:2]+'-'+date[2:]
    
    date = date if date[:2] in ['10','11','12'] else date[-4:]
    
    fullDate = d['startTime'][()][:8]
    titleDate = datetime.strptime(fullDate, '%Y%m%d').strftime('%A %B %d, %Y')
     
    
    directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Masking\active_mice'
    dataDir = os.path.join(os.path.join(directory, mouse_id), 'Plots/') 
    
    outfileDir = ('/' + mouse_id + '/Daily Summary/')
    outfileName = (mouse_id + ' Daily Summary ' + date + '.pdf')
    outfilePath = os.path.join(outfileDir, outfileName)
    
    c = canvas.Canvas(directory + outfilePath, pagesize=letter)

# page 1    
# insert "Daily Summary" at top bold, mouse id, and date 
    c.setFont('Helvetica-Bold', 12)
    c.drawString(2*inch, 10.5*inch, 'Daily Summary:   ')
    c.setFont('Helvetica', 12)
    c.drawString(3.3*inch, 10.5*inch, mouse_id + '                 ' + titleDate)
    
    
    
    
# insert daily wheel plot on left of canvas
    reportlab.platypus.Image(dataDir + '/Wheel Plots/Daily Wheel/' + mouse_id + ' ' + date + '.png', 
                             width=6*inch, height=4.5*inch).drawOn(c, .1*inch, 5.5*inch)
    
# no response wheel plot under daily wheel, same size
    reportlab.platypus.Image(dataDir + '/Wheel Plots/No Resp Wheel/' + mouse_id + ' ' + date + ' no resp.png',
                             width=6*inch, height=4.5*inch).drawOn(c, .1*inch, .2*inch)
    
# insert textbox with daily summary to right of plot - use textObject 
# set text origin 6.25 from left, .9 from top (inches)
# use textLines and moveCursor
    sessionText = c.beginText()
    sessionText.setTextOrigin(6.2*inch, 9.8*inch)
    sessionText.setFont('Helvetica', 9)
    sessionText.setLeading(12)
    sessionText.setWordSpace(1)
    session = session_stats(d, returnAs='str_array')
    for stat in session:
        sessionText.textLine(stat)
    c.drawText(sessionText)
    
    noResponse = c.beginText()
    noResponse.setTextOrigin(6.2*inch, 3*inch)
    noResponse.setFont('Helvetica', 8.5)
    noResponse.textLine('No Response Trials')
    c.drawText(noResponse)
    
# break
    c.showPage()
    
    
# page 2    
# insert catch trial wheel trace plot top left 
    reportlab.platypus.Image(dataDir + '/Wheel Plots/Catch/' + mouse_id + ' catch ' + date + '.png',
                             width=6*inch, height=4.5*inch).drawOn(c, .2*inch, 6*inch)
    
# insert textbox to right of plot with catch trial counts
    catchText = c.beginText()
    catchText.setTextOrigin(6.2*inch, 9.8*inch)
    catchText.setFont('Helvetica', 9)
    catchText.setLeading(12)
    catchText.setWordSpace(1)
    catch = catch_trials(d, arrayOnly=True)
    for count in catch:
        catchText.textLine(count)
    c.drawText(catchText)
    
    
# insert session plot - takes up entire bottom half, is already perfect size 
    reportlab.platypus.Image(dataDir + '/Session plots/' + mouse_id + ' session ' + date + '.png',
                             width=8.2*inch, height=5.5*inch).drawOn(c, 0, 0) 
# break
    c.showPage()
    
    
# page 3
# insert frame dist plot in upper left 1/6 
    reportlab.platypus.Image(dataDir + '/Other plots/frame dist/' + 'frame dist ' + date + '.png',
                             width=4*inch, height=3*inch).drawOn(c, .2*inch, 7.5*inch)
    
# insert frame intervals plot in upper right 1/6 (same size)
    reportlab.platypus.Image(dataDir + '/Other plots/frame intervals/' + 'frame intervals ' + date + '.png',
                             width=4*inch, height=3*inch).drawOn(c, 4.2*inch, 7.5*inch)
    
# insert wheel pos dist plot underneath 
    reportlab.platypus.Image(dataDir + '/Other plots/wheel pos/' + 'wheel ' + date + '.png',
                             width=8*inch, height=6*inch).drawOn(c, .5*inch, 1*inch)    
# break
    c.showPage()
    
# page 4    
# insert qVio sum plot top left 2/3s
    
    reportlab.platypus.Image(dataDir + '/Other plots/quiescent violations/' + 'Qvio ' + date + '.png',
                             width=6*inch, height=4.5*inch).drawOn(c, .1*inch, 5.5*inch)
    
# insert textbox in top right 1/3 with Qvios and max
    violations = check_qviolations(d, arrayOnly=True)
    
    qvioText = c.beginText()
    qvioText.setTextOrigin(6.2*inch, 9.8*inch)
    qvioText.setFont('Helvetica', 9)
    qvioText.setLeading(12)
    qvioText.setWordSpace(1)
    
    for i in violations:
        qvioText.textLine(i)
    c.drawText(qvioText)
    
# insert qVio count below qVio sum, same size 
    reportlab.platypus.Image(dataDir + '/Other plots/quiescent violations/' + 'Qvio ' + date + ' count.png',
                             width=6*inch, height=4.5*inch).drawOn(c, .2*inch, .2*inch)
#finish pdf    
    c.showPage()
# save 
    c.save()

