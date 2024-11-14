#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Multi-scale oriented energy network (MSOE net)

author: 
Antonino Greco, PhD
   
"""

import numpy as np
import tensorflow as tf

def load_param():
    param = np.load('dyn_model/weights.npz', allow_pickle=1)
    return param

def contrast_norm(input_layer, eps=1e-12):
    mean, var = tf.nn.moments(input_layer, axes=[1, 2, 3, 4],keepdims=True)
    std = tf.sqrt(var + eps)
    return (input_layer - mean) / std
    
def gauss2d_kernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def blur_downsample3d(input_layer, kernel_spatial_size,spatial_stride, sigma=2):
    # gauss kernel
    w = tf.constant(gauss2d_kernel((kernel_spatial_size,
                                    kernel_spatial_size), sigma=sigma),
                    dtype=tf.float32)
    w = tf.reshape(w, [1, kernel_spatial_size, kernel_spatial_size, 1, 1])

    # spatially pad the image sequence, but not temporally
    input_layer = tf.pad(input_layer,
                         [[0, 0], [0, 0],
                          [int(kernel_spatial_size / 2),
                           int(kernel_spatial_size / 2)],
                          [int(kernel_spatial_size / 2),
                           int(kernel_spatial_size / 2)],
                          [0, 0]], 'SYMMETRIC')

    return tf.nn.conv3d(input_layer, w,
                        strides=[1, 1, spatial_stride, spatial_stride, 1],
                        padding='VALID')
     
def max_pool3d(input_layer, kernel_spatial_size,kernel_temporal_size, spatial_stride=1):
    return tf.nn.max_pool3d(input_layer,
                            ksize=[1, kernel_temporal_size,
                                   kernel_spatial_size,
                                   kernel_spatial_size, 1],
                            strides=[1, 1, spatial_stride,
                                     spatial_stride, 1],
                            padding='SAME')
   
def l1_normalize(input_layer, axis=4, eps=1e-12):
    abs_sum = tf.reduce_sum(tf.abs(input_layer), axis, keepdims=True)
    input_layer_inv_norm = tf.math.reciprocal(tf.maximum(abs_sum, eps))
    return tf.multiply(input_layer, input_layer_inv_norm)
    
def bilinear_resample(input_layer, output_shape):
    return tf.image.resize(input_layer, output_shape)

def bilinear_resample3d(input_layer, output_shape, axis=1):
    unpacked = tf.unstack(input_layer, axis=axis)
    for i in range(len(unpacked)):
        unpacked[i] = bilinear_resample(unpacked[i], output_shape)
    return tf.stack(unpacked, axis=axis)

def channel_concat3d(input_layer, axis=4):
    return tf.concat(axis=axis, values=input_layer)




class MSOEnet:
    def __init__(self):
        param = load_param()

        self.conv1_w = tf.convert_to_tensor(param['conv1_w'])      
        self.conv1_b = tf.convert_to_tensor(param['conv1_b'])      

        self.conv2_w = tf.convert_to_tensor(param['conv2_w'])      
        self.conv2_b = tf.convert_to_tensor(param['conv2_b'])      

        self.conv3_w = tf.convert_to_tensor(param['conv3_w'])      
        self.conv3_b = tf.convert_to_tensor(param['conv3_b'])      

        self.conv4_w = tf.convert_to_tensor(param['conv4_w'])      
        self.conv4_b = tf.convert_to_tensor(param['conv4_b'])      

    def __call__(self, x, output=['concat']):
        out = {}
        concat_msoes, msinput = self.pyramid_block(x)
        if 'concat' in output:
            out['concat'] = concat_msoes
        if 'decode' in output:
            out['decode'] =  self.decode_block(concat_msoes)[tf.newaxis,...]
        return out
    
    def MSOE(self, input_layer):
        """
        Construct the MSOE network graph structure
        """
        # first convolution (2x11x11x1x32)
        conv1 = self.conv3d(input_layer, self.conv1_w, self.conv1_b, 11,2,32)
        # activation
        h_conv1 = tf.square(conv1)
        # max pooling (1x5x5x1x1)
        pool1 = max_pool3d(h_conv1, 5, 1)
        # second convolution (1x1x1x32x64)
        pool1.shape
        conv2 = self.conv3d(pool1, self.conv2_w, self.conv2_b, 1, 1, 64)
        # channel-wise l1 normalization (batchx1xHxWx64)
        l1_norm = l1_normalize(conv2)
        return l1_norm
        
    def pyramid_block(self,input_layer, num_scales=5):
        input_layer = contrast_norm(input_layer)
        inputs = [input_layer]
        msoes = [self.MSOE(inputs[0])]
        for scale in range(1, num_scales):
            # downsample data (batchx2xhxwx1)
            inputs.append(blur_downsample3d(inputs[scale-1], 5, 2,sigma=2))
            # create a MSOEnet and insert downsampled data (batchx1xhxwx64)
            small_msoe = self.MSOE(inputs[scale])
            # upsample MSOEnet output to original input size (batchx1xHxWx64)
            msoe = bilinear_resample3d(small_msoe,tf.shape(inputs[0])[2:4])
            msoes.append(msoe)
            
        multiscale_inputs = [input[:, 0] for input in inputs]
         
        # channel concat msoe outputs
        concat_msoes = channel_concat3d(msoes)
        return concat_msoes, multiscale_inputs

    def decode_block(self,concat_msoes):
        # third convolution (1x3x3x64*num_scalesx64)
        conv3 = self.conv3d(concat_msoes,self.conv3_w, self.conv3_b, 3, 1, 64)
        
        # activation
        h_conv3 = tf.nn.relu(conv3)
        
        # fourth convolution (flow out i.e. decode) (1x1x1x64x2)
        output = self.conv3d(h_conv3, self.conv4_w, self.conv4_b, 1, 1, 2)
        
        # reshape (batch x H x W x 2)
        return tf.reshape(output, [-1, tf.shape(output)[2],tf.shape(output)[3], 2])       

    def conv3d(self, input_layer, weights, biases, kernel_spatial_size,kernel_temporal_size, out_channels):
        
        input_layer = tf.pad(input_layer,
                             [[0, 0], [0, 0],
                              [int(kernel_spatial_size / 2),
                               int(kernel_spatial_size / 2)],
                              [int(kernel_spatial_size / 2),
                               int(kernel_spatial_size / 2)],
                              [0, 0]], 'SYMMETRIC')
    
        conv_output = tf.nn.conv3d(input_layer, weights,
                                   strides=[1, 1, 1, 1, 1],
                                   padding='VALID')
        
        return tf.nn.bias_add(conv_output, biases)

