# -*- coding: utf-8 -*-
"""
Utilities functions

Copyright [2024-Present] [Antonino Greco & Markus Siegel]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
    
"""

import cv2 as cv
import numpy as np
import tensorflow as tf

resize    = lambda frame, shape: cv.resize(frame,shape,fx=0,fy=0, interpolation = cv.INTER_CUBIC)
bgr2rgb   = lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2RGB)
rgb2gray  = lambda frame: cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
rotate90c = lambda frame: cv.rotate(frame,cv.ROTATE_90_CLOCKWISE)


def tf_rgb2gray(rgb):
    m = np.ones(rgb.shape,dtype='float32')
    m[...,:] = [0.299, 0.587, 0.114]
    return tf.einsum('abcde,abcde->abcd',rgb, tf.convert_to_tensor(m))[...,tf.newaxis]


def import_video(fname,size=None):
    cap = cv.VideoCapture(fname)
    fps = cap.get(cv.CAP_PROP_FPS)
    frames = []
    while True:
        read, f = cap.read()
        if read: frames.append(f)
        else: break
    nframe = len(frames)
    if nframe==0: raise Exception('It seems that the path is wrong. No video was loaded') 
    if size is None: h,w = frames[0].shape[:2]  
    else: w,h= size        
    out = np.zeros((nframe,h,w,frames[0].shape[2]),dtype='uint8')
    for i in range(nframe):
        if size is not None: tmp = resize(frames[i],size)
        else: tmp = frames[i]
        out[i,...] = bgr2rgb(tmp)
    return out, fps


def resize_max(frames,max_dim=250):
    shape = frames.shape[1:3]
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = int(shape[0]*scale),int(shape[1]*scale)
    out = np.zeros((frames.shape[0],*new_shape,frames.shape[3]),dtype='uint8')
    for i in range(frames.shape[0]):
        out[i,...] = resize(frames[i,...],(new_shape[1],new_shape[0]))
    return out


def preprocess(frames):
    return (frames/255).astype('float32')


def deprocess(frames):
    return (frames*255).astype('uint8')


def save_video(frames,fname='out.mp4',fps=24):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    vid = cv.VideoWriter(fname,fourcc, fps, (frames.shape[2],frames.shape[1]))
    for i in range(frames.shape[0]):
        vid.write(cv.cvtColor(frames[i,...], cv.COLOR_RGB2BGR))
    vid.release()


