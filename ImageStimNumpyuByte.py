# -*- coding: utf-8 -*-
"""
Created on Mon May 09 17:10:06 2016

@author: labadmin
"""

import numpy
import pyglet
GL = pyglet.gl
from psychopy import visual

class ImageStimNumpyuByte(visual.ImageStim):

    '''Subclass of ImageStim which allows fast updates of numpy ubyte images,
       bypassing all internal PsychoPy format conversions.
    '''

    def __init__(self,
                 win,
                 image=None,
                 mask=None,
                 units="",
                 pos=(0.0, 0.0),
                 size=None,
                 ori=0.0,
                 color=(1.0, 1.0, 1.0),
                 colorSpace='rgb',
                 contrast=1.0,
                 opacity=1.0,
                 depth=0,
                 interpolate=False,
                 flipHoriz=False,
                 flipVert=False,
                 texRes=128,
                 name='', autoLog=True,
                 maskParams=None):

        if image is None or type(image) != numpy.ndarray or len(image.shape) != 2:
            raise ValueError(
                'ImageStimNumpyuByte must be numpy.ubyte ndarray (0-255)')

        self.interpolate = interpolate

        # convert incoming Uint to RGB trio only during initialization to keep PsychoPy happy
        # else, error is: ERROR   numpy arrays used as textures should be in
        # the range -1(black):1(white)

        data = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.float32)
        # (0 to 255) -> (-1 to +1)
        fimage = image.astype(numpy.float32) / 255 * 2.0 - 1.0
        k = fimage[0, 0] / 255
        data[:, :, 0] = fimage#R
        data[:, :, 1] = fimage#G
        data[:, :, 2] = fimage#B

        visual.ImageStim.__init__(self,
                                  win,
                                  image=data,
                                  mask=mask,
                                  units=units,
                                  pos=pos,
                                  size=size,
                                  ori=ori,
                                  color=color,
                                  colorSpace=colorSpace,
                                  contrast=contrast,
                                  opacity=opacity,
                                  depth=depth,
                                  interpolate=interpolate,
                                  flipHoriz=flipHoriz,
                                  flipVert=flipVert,
                                  texRes=texRes,
                                  name=name, autoLog=autoLog,
                                  maskParams=maskParams)

    def setReplaceImage(self, tex):
        '''
        Use this function instead of 'setImage' to bypass format conversions
        and increase movie playback rates.
        '''
        #intensity = tex.astype(numpy.ubyte)
        intensity = tex
        internalFormat = GL.GL_LUMINANCE
        pixFormat = GL.GL_LUMINANCE
        dataType = GL.GL_UNSIGNED_BYTE
        # data = numpy.ones((intensity.shape[0],intensity.shape[1],3),numpy.ubyte)#initialise data array as a float
        # data[:,:,0] = intensity#R
        # data[:,:,1] = intensity#G
        # data[:,:,2] = intensity#B
        data = intensity
        texture = tex.ctypes  # serialise
        try:
            tid = self._texID  # psychopy renamed this at some point.
        except:
            tid = self.texID
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        # makes the texture map wrap (this is actually default anyway)
        if self.interpolate:
            interpolation = GL.GL_LINEAR
        else:
            interpolation = GL.GL_NEAREST
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER, interpolation)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER, interpolation)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internalFormat,
                        # [JRG] for non-square, want data.shape[1], data.shape[0]
                        data.shape[1], data.shape[0], 0,
                        pixFormat, dataType, texture)
        pass

if __name__ == "__main__":
    pass