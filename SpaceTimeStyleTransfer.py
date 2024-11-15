# -*- coding: utf-8 -*-
"""
SpaceTime Style Transfer (STST)

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

import os
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Color_Transfer import color_transfer
from SpaceTimeExtractor import SpaceExtractor, TimeExtractor
from utils import deprocess, save_video


class SpaceTimeStyleTransfer:
    
    def __init__(self, targets):

        self.lossnames_short = ['sc','ss','tc','ts']
        self.lossnames_long = ['space_content','space_style','time_content','time_style']        
        self.lossnames = {}
        for long, short in zip(self.lossnames_long, self.lossnames_short):
            self.lossnames[long] = short
            self.lossnames[short] = long
            
        # =====================================================================
        # Parse input
        # =====================================================================

        if not isinstance(targets, dict): 
            raise Exception('Please specify the target videos as a dictionary '+
                            'where the keys are one of the 4 different losses')

        for k in targets.keys(): 
            if k not in self.lossnames.keys():
                raise Exception('Unknown inserted Loss name %s'%(k))

        if not all([isinstance(x,np.ndarray) for x in targets.values()]): 
            raise Exception('Please specify at least one video in targets as 4D'+
                            ' numpy arrays. If multiple, targets must be a list')
        
        if not all([(x>=0).all() and (x<=1).all() for x in targets.values()]):
            raise Exception('All input videos must have values in the range [0,1].'+
                            ' There is at least one of the input videos that is not in [0,1].')

        if not len(set([x.shape for x in targets.values()]))==1:
            raise Exception('All input videos must have same shape. Please '+
                            'check the shape of the input videos.')


        self.targets = {}
        for k,v in targets.items():
            
            losskey = k if len(k) < 3 else self.lossnames[k]
            
            self.targets[losskey] = v.astype('float32')
        
        tar_id_unique = np.unique([v.__array_interface__['data'][0] for v in targets.values()])
        self.tar_loss_unique = []
        for idx in tar_id_unique:            
            for k,v in targets.items(): 
                if v.__array_interface__['data'][0] == idx:
                    self.tar_loss_unique.append(k)
                    break
                    
        self.tar_id = {k:np.where(v.__array_interface__['data'][0] == tar_id_unique)[0][0] for k,v in targets.items()}
        self.n_targets = len(tar_id_unique)
        self.shape = list(self.targets.values())[0].shape
        self.losskeys = list(self.targets.keys())
        self.loss_sel = {'space': 'sc' in self.losskeys or 'ss' in self.losskeys,
                         'time': 'tc' in self.losskeys or 'ts' in self.losskeys}

        for k in self.lossnames_short:
            self.loss_sel[k] = k in self.losskeys

        
        
        default_layers = dict(content_layers=None,style_layers=None)

        if self.loss_sel['space']:
            self.SpaceExtractor = SpaceExtractor(**default_layers)

        if self.loss_sel['time']:
            self.TimeExtractor = TimeExtractor(**default_layers)
        
        self.nlayers = {}
        for k in self.losskeys:
            ext = self.SpaceExtractor if k[0] == 's' else self.TimeExtractor
            n = ext.content_nlayers if k[1] == 'c' else ext.style_nlayers
            self.nlayers[k] = n
            
                     
        self.config_default = dict(
            opt_settings = [self.opt_settings_default()], 
            opt_info = dict(verbose=100, verbose_plot=False),
            hyperparams = dict(device='/GPU:0', batch=1, blend=.9, pad=5, col_trans=['ss'], equalize_losses=1e4),
            out_settings = dict(fname='stst', fps=25)            
            )
                
        
    def check_config(self, config):
        
        print('-'*25 + '\nCheck configuration\n')

        if not isinstance(config, dict):
            raise Exception('Config must be a dict')

        for k in config.keys():
            if k not in self.config_default.keys():
                print('The inserted | %s | argument is not a valid field, will be ignored'%(k))
                            
                
        for k,v in self.config_default.items():
            if k not in config.keys():
                print('| %s | argument not found, using default'%(k))
                config[k] = v
            
            else:
                if k != 'opt_settings':
                    
                    if not isinstance(config[k], dict):
                        print('The inserted %s argument is not a dict, using default'%(k))
                        config[k] = v
                    else:
                    
                        for kk, vv in self.config_default[k].items():
                            if kk not in config[k].keys():
                                print('| %s | -> | %s | field not found, using default'%(k, kk))
                                config[k][kk] = vv
                                
                else:

                    if not isinstance(config[k], list):
                        raise Exception('Config[\'%s\'] must be a list'%(k))
                    
                    if isinstance(config[k][0], list):            
                        
                        if len(config[k]) < self.shape[0]:
                            raise Exception('If you define %s for each frame, then it must match at least the number of target frames')
                        
                        loss_w_keys = list(config[k][0][0]['loss_w'].keys()) 
                        reg_w_keys = list(config[k][0][0]['regularizer_w'].keys())
                        for f, frame in enumerate(config[k]):
                            if not isinstance(frame, list):
                                raise Exception('%s at frame %d is not a list'%(k,f+1))
                                
                            for o, octave in enumerate(frame):                                
                                for k_oct, v_oct in self.config_default[k][0].items():
                                    if k_oct not in octave.keys():
                                        print('| %s | -> | frame %d | -> | octave %d | -> | %s | not found, using default'%(k, f+1, o+1, k_oct))
                                        config[k][o][k_oct] = v_oct
                                        
                                for k_loss_w in loss_w_keys:
                                    if k_loss_w not in octave['loss_w'].keys():
                                        config[k][o]['loss_w'][k_loss_w] = 0

                                for k_reg_w in reg_w_keys:
                                    if k_reg_w not in octave['regularizer_w'].keys():
                                        config[k][o]['regularizer_w'][k_reg_w] = 0
                                
                                    
                    else:
                        
                        for o, octave in enumerate(config[k]):                                
                            for k_oct, v_oct in self.config_default[k][0].items():
                                if k_oct not in octave.keys():
                                    print('| %s | -> | octave %d | -> | %s | not found, using default'%(k, o+1, k_oct))
                                    config[k][o][k_oct] = v_oct
                                else:
                                    if isinstance(v_oct, dict):
                                        if not isinstance(octave[k_oct], dict):
                                            raise Exception('%s at octave %d is supposed to be a dict'%(k_oct,o))                                        

                                        for k_param, v_param in v_oct.items():
                                            if k_param not in octave[k_oct].keys():
                                                octave[k_oct][k_param] = v_param
                                            


        if config['hyperparams']['pad'] < 1: 
            print('The number of frames for padding is not valid (less than one), using one')
            config['hyperparams']['pad'] = 1

        if config['hyperparams']['batch'] < 1: 
            print('The batch is not valid (less than one), using one')
            config['hyperparams']['batch'] = 1

        if not isinstance(config['hyperparams']['col_trans'], list):
            raise Exception('Color transfer hyperparameter must be a list. '+
                            'Set an empty list for unselecting this option')
        else:
            if len(config['hyperparams']['col_trans']) > 1:
                print('Color transfer hyperparameter is more than one, using first')
                config['hyperparams']['col_trans'] = config['hyperparams']['col_trans'][:1]
            
            if config['hyperparams']['col_trans'][0] not in self.losskeys:     
                raise Exception('Color transfer hyperparameter must be one of the inserted targets')

        self.regkeys = list(config['opt_settings'][0]['regularizer_w'].keys())

        print('\nCheck completed\n' + '-'*25 + '\n')

        return config
        
        
    def opt_settings_default(self):        
        return dict(            
            optimizer = dict(func='SGD', normalize_grad=True, params=dict(learning_rate=.005)),
            iters=100,
            scale = dict(octave_scale=1.5, octave=0),
            regularizer_w = {'tv':1e-1},
            loss_w = {k:v for k,v in zip(self.losskeys, [1 for _ in self.losskeys])}
            )

        
    def get_opt_settings(self, config, f):        
        if isinstance(config['opt_settings'][0], dict):
            return config['opt_settings']
        else:
            return config['opt_settings'][f]

            
    def FeatureExtractor(self, inputs):
        
        if not isinstance(inputs, dict):
            inputs = {k:inputs for k in self.losskeys}                    
        
        # Initialize output
        feat = {}
        
        for k, frames in inputs.items():
            
            if k[0] == 's': 

                if len(frames.shape) != 4:
                    space_inputs = frames[:,1,:,:,:]
                else:
                    space_inputs = frames
                        
                feat[k] = self.SpaceExtractor(space_inputs, feat=k[1]) 

            elif k[0] == 't':                
                feat[k] = self.TimeExtractor(frames, feat=k[1]) 
        
        return feat


    def L2_loss(self, y_true, y_pred):
        return tf.reduce_mean( (y_pred - y_true)**2 ) 
        

    def loss_fn(self, gen, feat, loss_w, reg_w, feat_tar=None):

        if feat_tar is None:
            
            feat_gen, feat_tar = {}, {}
            for k,v in feat.items():
                if k[1] == 's': # style
                    feat_gen[k] = [[x[1:] for x in v[0]], [x[1:] for x in v[1]]]
                    feat_tar[k] = [[x[:1] for x in v[0]], [x[:1] for x in v[1]]]
                else:
                    feat_gen[k] = [x[1:] for x in v]
                    feat_tar[k] = [x[:1] for x in v]
                    
        else:
            feat_gen = feat

        losses = {}        
        for k in self.losskeys:
            if k[1] == 's':
                
                lays_gen, lays_tar = feat_gen[k][1], feat_tar[k][1]
                losses[k] = tf.add_n( [self.L2_loss(act_gen, act_tar) 
                                       for act_gen, act_tar in zip(lays_gen, lays_tar)] )            
    
                losses[k] *=  tf.cast(loss_w[k]/self.nlayers[k], tf.float32)
                 
                
            else: 
                
                lays_gen, lays_tar = feat_gen[k], feat_tar[k]
    
                losses[k] = tf.add_n( [self.L2_loss(act_gen, act_tar) 
                                       for act_gen, act_tar in zip(lays_gen, lays_tar)] )            
    
                losses[k] *=  tf.cast(loss_w[k]/self.nlayers[k], tf.float32)

        for k in self.regkeys:        

            if k=='tv':
                vol = tf.cast(gen.shape[2]*gen.shape[3], tf.float32)
                losses[k] = reg_w[k] * tf.concat([ (tf.image.total_variation(gen[b,1:,...])/vol)[tf.newaxis]  for b in range(gen.shape[0])], axis=0)

        
        losskeys = list(losses.keys())
        loss = losses[losskeys[0]]
        for i in range(1,len(losskeys)): 
            loss += losses[losskeys[i]]
        
        return loss, losses


    @tf.function
    def optimize(self, gen, tar, loss_w, reg_w, feat_tar=None):
        
        with tf.device(self.config['hyperparams']['device']):
            
            with tf.GradientTape() as tape:
                tape.watch(gen)
                
                if feat_tar is None:
                    inputs = self.concatenate_tar_gen(tar,gen)
                else:
                    inputs = gen
                
              
                feat = self.FeatureExtractor(inputs)

                loss, losses = self.loss_fn(gen, feat, loss_w, reg_w, feat_tar)

            grad = tape.gradient(loss, gen)
            
        return loss, losses, grad   


    def update_params(self, optim, gen, grad, clip=True):
        
        grad = grad[:,1:,...]

        if optim['normalize_grad']:
            std = tf.math.reduce_std(grad) + 1e-8 
            grad = grad / std

        batch, frames, h, w, c = gen.shape
        grad = tf.concat((tf.zeros((batch, 1, grad.shape[2], grad.shape[3], c)), grad), axis=1)

            
        lr = optim['params']['learning_rate']
        gen -= lr*grad

        if clip:
            gen = tf.clip_by_value(gen, clip_value_min=0.0, clip_value_max=1.0)

        return gen   


    def generate(self, config):
        
        self.config = self.check_config(config)

        verbose = self.config['opt_info']['verbose']
        batch = self.config['hyperparams']['batch']
        pad = self.config['hyperparams']['pad']        
        nframe = self.shape[0]
        nframe_padded = pad + nframe

        pre_frame = tf.random.uniform((batch, 1, *self.shape[1:]))
        cur_frame = tf.random.uniform((batch, 1, *self.shape[1:]))
        
        self.pad_frames_tar()
        
        self.out = np.zeros((batch, *self.shape), dtype='float32')
        self.history = {k:[[] for _ in range(nframe_padded)] for k in self.losskeys + self.regkeys}
        self.elapsed_time = []

        self.print_summary()
        
        if verbose: self.verbose('\nEstimate finish time')
        t0 = time.time()
        eta = self.estimate_eta(0)
        if verbose: self.verbose('\n\nEstimated finish time is %s\n\nStarting optimization process\n'%(eta)+'.'*25 + '\n')

        
        for f in range(1, nframe_padded): 

            frame_t0 = time.time()
            if verbose: 
                msg = '\nFrames transition %d -> %d out of %d... Estimated finishing in %s'%(
                        f - pad, f - pad + 1, nframe, eta)
                self.verbose(msg)

            gen = tf.concat((pre_frame, cur_frame), axis=1)


            tar = self.extract_frames_from_tar(f)
                        
            opt_settings = self.get_opt_settings(self.config, f)            
            
            gen = self.optimize_frame_transition(f, gen, tar, opt_settings)

            gen_frame, pre_frame, cur_frame = self.postprocess(f, gen)
            
            if f - pad >= 0:
                self.out[:, f - pad, ...] = gen_frame

            frame_time = time.time() - frame_t0
            eta = self.estimate_eta(f, frame_time)
            
        if verbose:
            eta = time.time()-t0
            msg = "\nElapsed time was %1.1f minutes (%1.1f hours)"%(eta/60,eta/3600)
            self.verbose('\n'+msg)
                
        self.save_results()
            
        return self.out                


    def optimize_frame_transition(self, f, gen, tar, opt_settings):
        
        ti = time.time()
        
        for o, octave in enumerate(opt_settings):
            
            iters = octave['iters']
            optim = octave['optimizer']
            loss_w = octave['loss_w']
            reg_w = octave['regularizer_w']
            
            
            gen, tar, oct_shape = self.prepare_octave(gen, tar, octave)
                            
            feat_tar = self.FeatureExtractor(tar) 
                
            if self.config['hyperparams']['equalize_losses']:
                loss_w, reg_w = self.equalize_loss_weights(gen, tar, loss_w, reg_w, feat_tar)

                
            for i in range(iters):
                loss, losses, grad = self.optimize(gen, tar, loss_w, reg_w, feat_tar)  
                                    
                gen = self.update_params(optim, gen, grad)
                self.update_history(f, losses)
                ti = self.print_info(i, f, octave, ti, gen)
        
        return gen


    def equalize_loss_weights(self, gen, tar, loss_w, reg_w, feat_tar):

        equalize_val = self.config['hyperparams']['equalize_losses']
        loss_w_eq, reg_w_eq = {}, {}
        loss_w_baseline = {k:1 for k in loss_w.keys()}
        reg_w_baseline = {k:1 for k in reg_w.keys()}
        loss, losses, grad = self.optimize(gen, tar, loss_w_baseline, reg_w_baseline, feat_tar) 

        for k,v in losses.items():

            if k in loss_w.keys():
                loss_w_eq[k] = tf.convert_to_tensor(loss_w[k]*(equalize_val/v))

            if k in reg_w.keys():
                reg_w_eq[k] = tf.convert_to_tensor(reg_w[k]*(equalize_val/v))

        return loss_w_eq, reg_w_eq


    def postprocess(self, f, gen):
        
        if gen.shape[2] != self.shape[1] or gen.shape[3] != self.shape[2]:
            gen = tf.concat([tf.image.resize(gen[b], self.shape[1:-1])[tf.newaxis] for b in range(gen.shape[0])], axis=0)  
        
        gen_frame = gen[:,1:,...].numpy()

        col_trans = self.config['hyperparams']['col_trans']
        if col_trans:            
            tar = self.targets[col_trans[0]][f,...]            
            gen_frame = [color_transfer(gen_frame[b,0], tar)[np.newaxis][np.newaxis] for b in range(gen_frame.shape[0])]
            gen_frame = np.concatenate(gen_frame, axis=0).astype('float32')

            if self.config['opt_info']['verbose_plot']:
                self.plot_frames(gen, 'Frame %d - Color trasfered'%f)
                    
                
        pre_frame = gen_frame.copy()

        blend = self.config['hyperparams']['blend']
        cur_frame = blend*pre_frame + (1-blend)*np.random.rand(gen.shape[0], 1, *self.shape[1:]).astype('float32')

        gen_frame = np.squeeze(gen_frame)

        return gen_frame, pre_frame, cur_frame


    def estimate_eta(self, f, frame_time=None):

        nframe_padded = self.config['hyperparams']['pad'] + self.shape[0] 

        # Check whether ETA has to be esitmated for the first time
        if f==0:
                        
            t0 = time.time()

            gen = tf.random.uniform((self.config['hyperparams']['batch'], 2, *self.shape[1:]))

            # Get current frame transition from targets' videos
            tar = self.extract_frames_from_tar(f+1)
                        
            # Get current optimization settings
            opt_settings = self.get_opt_settings(self.config, f+1)     

            # Initialize elapsed time
            frame_time = time.time() - t0

            # Run every octave indipendently to estimate elapsed time            
            for i in range(len(opt_settings)):                

                # Set a limited number of iters to estimate ETA
                actual_iters = int(opt_settings[i]['iters'])
                tmp_opt = opt_settings[i].copy()
                tmp_opt['iters'] = 10
                
                t0 = time.time()
                self.optimize_frame_transition(f+1, gen, tar, [tmp_opt])
                
                iter_time = (time.time() - t0)/opt_settings[i]['iters']
                octave_time = iter_time * (actual_iters - 1)
                frame_time += octave_time
                
            eta = frame_time * nframe_padded
                
        else:
            
            if self.config['opt_info']['verbose']: 
                self.verbose('Elapsed time for this frame was %.1f sec (%.1f min)'%(
                              frame_time, frame_time/60))

            self.elapsed_time.append(frame_time)
            eta = (nframe_padded - f)*(np.median(self.elapsed_time))

        h, s =  eta // 3600, eta % 3600
        m, s = s // 60, s % 60
        return '%d:%d:%d'%(h,m,s)


    def extract_frames_from_tar(self, f):
        return {k:tf.convert_to_tensor(v[f-1:f+1][np.newaxis]) for k,v in self.targets.items()}
        
    
    def pad_frames_tar(self):
        # Mirror padding
        pad = self.config['hyperparams']['pad']     

        for k in self.targets.keys():
            tmp = np.zeros((pad, *self.shape[1:]), dtype='float32')
            
            for i in reversed(range(1,pad+1)):
                tmp[i-1] = self.targets[k][i]

            self.targets[k] = np.concatenate((tmp,self.targets[k]))           

    
    def concatenate_tar_gen(self, tar, gen):        
        return {k:tf.concat((tar[k],gen), axis=0) for k in self.losskeys}
        
    
    def prepare_octave(self, gen, tar, octave):
        octave_pow = octave['scale']['octave']
        octave_scale = octave['scale']['octave_scale']
        oct_shape = tf.cast(tf.cast(self.shape[1:-1],tf.float32) * octave_scale**octave_pow, tf.int32)
        
        gen = tf.concat([tf.image.resize(gen[b], oct_shape)[tf.newaxis] for b in range(gen.shape[0])], axis=0)                       
        tar = {k:tf.image.resize(tar[k][0], oct_shape)[tf.newaxis] for k in tar.keys()}                    
        
        return gen, tar, oct_shape


    def verbose(self, msg):
        print(msg)
        
        
    def print_info(self, i, f, octave, ti, gen):
        
        verbose = self.config['opt_info']['verbose']
        verbose_plot = self.config['opt_info']['verbose_plot']
        pad = self.config['hyperparams']['pad']
        iters = octave['iters']
        octave_pow = octave['scale']['octave']
        
        loss_val = sum([v[f][-1] for v in self.history.values()])
        
        # Print info about the optimization process
        if verbose and (i%verbose==0 or i==iters-1):
            msg = 'Frames %d -> %d | Oct %d | Iter %d | ETA: %1.1f s | Loss:%1.2f'%(
                f - pad, f - pad + 1, octave_pow, i, time.time() - ti, loss_val)
            
            for k,v in self.history.items(): 
                msg += ' | '+k+': '+'%1.1f'%(v[f][-1])

            self.verbose(msg)

            ti = time.time()
            
            if verbose_plot:                
                title = 'Frames %d -> %d | Oct %d | Iter %d'%(f - pad, f - pad + 1, octave_pow, i)
                self.plot_frames(gen, title)
                
        return ti


    def print_summary(self):
        verbose = self.config['opt_info']['verbose']
        opt = self.config['opt_settings']
        
        # Print info    
        title = '#'*25 +'\nSpace Time Style Transfer\n'+'#'*25+'\n\n'

        info = 'Input videos are %d with shape %s. '%(self.n_targets, self.shape)

        if isinstance(opt[0], list):
            info += 'Losses weights are changing across frames\n'

        else:
            info += 'Losses weights are:\n'                
            for k, v in opt[0]['loss_w'].items():
                info += '%s:  %f  |  '%(k,v)
            info = info[:-5] + '\n\n'

        info += 'Hyper-parameters:\n'
        for k,v in self.config['hyperparams'].items():            
            info += '%s:  %s  |  '%(k,str(v))
        info = info[:-5] + '\n\n'
            
        info += 'Octaves:\n'
        
        for octave in opt:
            tmp = dict(lr=octave['optimizer']['params']['learning_rate'], iters=octave['iters'], 
                       **octave['scale'], **octave['regularizer_w'])
            
            for k,v in tmp.items():            
                info += '%s:  %s  |  '%(k,str(v))
            info = info[:-5] + '\n'
                        
        date = '\nOptimization process started at ' + datetime.datetime.now().strftime("%H:%M %d-%m-%Y")+'\n'+'.'*48
        if verbose: self.verbose(title+info+date)
        
        return info      


    def plot_frames(self, gen, title=''):
        batch = gen.shape[0]
        col = int(np.ceil(batch/2))
        row = col if col%2==0 else col -1

        plt.figure()
        if batch > 1:
            for b in range(batch):
                plt.subplot(row,col, b+1)
                plt.imshow(gen[b,1,...])                    
            plt.suptitle(title)
        else:
            plt.imshow(gen[0,1,...])
            plt.title(title)

        plt.show()
        plt.pause(.001)        
        
        
    def update_history(self, f, losses):
        for k,v in losses.items():
            # val =float(np.squeeze(v.numpy()))
            val =float(np.mean(v.numpy()))
            if k in self.history.keys():
                self.history[k][f].append(val)            
                
            else:
                self.history[k] = [[] for _ in range(self.config['hyperparams']['pad'] + self.shape[0])] 
                self.history[k][f].append(val) 

    
    def save_results(self):
        
        fname = self.config['out_settings']['fname']
        fps = self.config['out_settings']['fps']
        now = datetime.datetime.now().strftime("%y%m%d_%H%M")

        # Save video in mp4 format
        for b in range(self.out.shape[0]):
            self.save_video(fname + '_b%d'%(b), self.out[b], fps, now)


    def save_video(self, fname, frames, fps, now=None):
        frames = deprocess(frames)
        if now is None: now = datetime.datetime.now().strftime("%y%m%d_%H%M")
        if fname is None: fname = 'output'
        # title = os.path.basename(fname)
        # path = os.path.dirname(fname)
        title = fname
        path = 'out'
        save_video(frames, now+'_'+title+'.mp4', fps=fps)

