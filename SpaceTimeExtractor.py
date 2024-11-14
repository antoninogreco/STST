# -*- coding: utf-8 -*-
"""
SpaceTime Extractors

author: 
Antonino Greco, PhD
    
"""

import numpy as np
import tensorflow as tf
from dyn_model.MSOEnet import MSOEnet



class SpaceExtractor(tf.keras.models.Model):
    
    def __init__(self, style_layers=None, content_layers=None, mom_order=5):
        
        super().__init__()
        
        # Define default style and content layers
        self.content_layers_default = ['block5_conv2'] 
        self.style_layers_default   = ['block1_conv1', 'block2_conv1',
                                       'block3_conv1', 'block4_conv1', 
                                       'block5_conv1']

        self.mom_order = mom_order

        self.style_layers = style_layers if style_layers is not None else self.style_layers_default
        self.style_nlayers = len(self.style_layers)
        self.content_layers = content_layers if content_layers is not None else self.content_layers_default
        self.content_nlayers = len(self.content_layers)
        
        # Load the pretrained VGG19 model
        self.model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        self.content_extractor = self.create_extractor(self.content_layers)
        self.style_extractor = self.create_extractor(self.style_layers)
            

    def create_extractor(self, out_layers):
        outputs = [self.model.get_layer(name).output for name in out_layers]        
        model = tf.keras.Model([self.model.input], outputs)
        model.trainable = False
        return model


    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)


    def call(self, inputs, feat='b'):
        
        inputs = inputs*255.0

        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        if feat=='b':
            content_outputs = self.content_extractor(preprocessed_input)
            style_activations = self.style_extractor(preprocessed_input)
            style_outputs = [self.gram_matrix(style_output) # Compute Gram matrices
                              for style_output in style_activations]
            
            if not isinstance(content_outputs, list): content_outputs = [content_outputs]
            if not isinstance(style_outputs, list): style_outputs = [style_outputs]
            return {'content': content_outputs, 'style': style_outputs}
        
        elif feat=='s':            
            style_activations = self.style_extractor(preprocessed_input)
            style_outputs = [style_activations,
                             [self.gram_matrix(style_output) # Compute Gram matrices
                             for style_output in style_activations]]
            
            

            if not isinstance(style_outputs, list): style_outputs = [style_outputs]
            return style_outputs
        
        elif feat=='c':
            content_outputs = self.content_extractor(preprocessed_input)
            if not isinstance(content_outputs, list): content_outputs = [content_outputs]
            return content_outputs
        

    def get_layers_output(self, layer_names):
        outputs = [self.model.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([self.model.input], outputs)
        return model

class TimeExtractor(tf.keras.models.Model):
    
    def __init__(self, style_layers=None, content_layers=None, mom_order=5): 

        super().__init__()

        self.content_layers_default = ['concat'] 
        self.style_layers_default   = ['concat']
        
        self.mom_order = mom_order

        self.style_layers = style_layers if style_layers is not None else self.style_layers_default
        self.style_nlayers = len(self.style_layers)
        self.content_layers = content_layers if content_layers is not None else self.content_layers_default
        self.content_nlayers = len(self.content_layers)

        self.model = MSOEnet()
        
        self.w_gray = np.zeros((3,3,3,1)).astype(np.float32)
        self.w_gray[1,1,:,:] = 1
        self.w_gray = tf.convert_to_tensor(self.w_gray)

        
    def gram_matrix(self,input_tensor):
        result = tf.linalg.einsum('bnijc,bnijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2]*input_shape[3], tf.float32)
        return result/(num_locations)


    def convert_to_grayscale(self, inputs):
        return tf.nn.conv2d(inputs, self.w_gray, (1,1), padding='SAME')
    

    def call(self, inputs, feat='b'): 
        """
        
        Parameters
        ----------
        inputs : tf.Tensor
            tensor with shape (N x F x H x W x C) and pixel in the range [0,1]
            if C = 3, the model convert the input to grayscale

        feats : string
            Select which feature to extract. Select 's' for style , 'c' for 
            content and 'b' for both content and style. The default is 'b'.

        Returns
        -------
        dict
            Content and Style features 

        """
        
        # Check whether inputs has 5 dim, othterwise assume it's missing the batch
        # one, i.e. the first one
        if len(inputs.shape) < 5:
            inputs = inputs[tf.newaxis]

        # Check whether the color format is RGB, in case convert to gray         
        if inputs.shape[-1] > 1:
            inputs = self.convert_to_grayscale(inputs)

        if feat == 'b':
            content_outputs = self.model(inputs,self.content_layers)
            content_outputs = [content_outputs[k] for k in self.content_layers]
            style_activations = self.model(inputs,self.style_layers)
            style_outputs = [self.gram_matrix(style_activations[k]) for k in self.style_layers] 

            return {'content': content_outputs, 'style': style_outputs}    
    
        elif feat == 's':            
            style_activations = self.model(inputs,self.style_layers)
            style_outputs = [[style_activations[k] for k in self.style_layers], [self.gram_matrix(style_activations[k]) for k in self.style_layers]] 
            return style_outputs
        
        elif feat == 'c':
            content_outputs = self.model(inputs,self.content_layers)
            content_outputs = [content_outputs[k] for k in self.content_layers]
            return content_outputs









