# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:47:32 2016

@author: Gale
"""


try:
    from PyQt4 import QtGui as Qt
except:
    from PyQt5 import QtWidgets as Qt

import h5py
import numpy as np


def getFile(caption='Choose File',rootDir='',fileType=''):
    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication([])
    output = Qt.QFileDialog.getOpenFileName(None,caption,rootDir,fileType)
    filePath = output[0] if isinstance(output,tuple) else output
    return str(filePath)


def getFiles(caption='Choose File',rootDir='',fileType=''):
    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication([])
    output = Qt.QFileDialog.getOpenFileNames(None,caption,rootDir,fileType)
    filePaths = output[0] if isinstance(output,tuple) else output
    return [str(f) for f in filePaths]

    
def getDir(caption='Choose Directory',rootDir=''):
    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication([])
    dirPath = Qt.QFileDialog.getExistingDirectory(None,caption,rootDir)
    return str(dirPath)
    

def saveFile(caption='Save As',rootDir='',fileType=''):
    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication([])
    output = Qt.QFileDialog.getSaveFileName(None,caption,rootDir,fileType)
    filePath = output[0] if isinstance(output,tuple) else output
    return str(filePath)


def objToHDF5(obj,filePath=None,fileOut=None,grp=None,saveDict=None,append='False'):
    if fileOut is None:
        if filePath is None:
            filePath = saveFile(fileType='*.hdf5')
            if filePath=='':
                return
        fileOut = h5py.File(filePath,'a')
        newFile = fileOut
    else:
        newFile = None
    if grp is None:    
        grp = fileOut['/']
    if saveDict is None:
        saveDict = obj.__dict__
    for key in saveDict:
        if key[0]=='_':
            continue
        elif type(saveDict[key]) is dict:
            objToHDF5(obj,fileOut=fileOut,grp=grp.create_group(key),saveDict=saveDict[key])
        else:
            try:
                grp.create_dataset(key,data=saveDict[key],compression='gzip',compression_opts=1)
            except:
                try:
                    grp[key] = saveDict[key]
                except:
                    try:
                        grp.create_dataset(key,data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
                    except:
                        print('Could not save: ', key)                  
    if newFile is not None:
        newFile.close()
                
                
def hdf5ToObj(obj,filePath=None,grp=None,loadDict=None):
    if grp is None:
        if filePath is None:        
            filePath = getFile(fileType='*.hdf5')
            if filePath=='':
                return
        grp = h5py.File(filePath,'r')
        newFile = grp
    else:
        newFile = None
    for key,val in grp.items():
        if isinstance(val,h5py._hl.dataset.Dataset):
            v = val.value
            if isinstance(v,np.ndarray) and v.dtype==np.object:
                v = v.astype('U')
            if loadDict is None:
                setattr(obj,key,v)
            else:
                loadDict[key] = v
        elif isinstance(val,h5py._hl.group.Group):
            if loadDict is None:
                setattr(obj,key,{})
                hdf5ToObj(obj,grp=val,loadDict=getattr(obj,key))
            else:
                loadDict[key] = {}
                hdf5ToObj(obj,grp=val,loadDict=loadDict[key])
    if newFile is not None:
        newFile.close()
                
                
if __name__=="__main__":
    pass 