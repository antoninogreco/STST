# -*- coding: utf-8 -*-
"""
Synthetize metamer videos

author: 
Antonino Greco, PhD

"""

import argparse
import tensorflow as tf



from utils import import_video, preprocess, resize_max
from SpaceTimeStyleTransfer import SpaceTimeStyleTransfer

parser = argparse.ArgumentParser(description="Synthesize videos with STST")

parser.add_argument('-gpu', help='which GPU to use as index', required=True)
parser.add_argument('-tar', help='file name of the target video, without the file format (i.e., ".mp4"). The format has to be MP4.', required=True)
parser.add_argument('-resize', help='resize target video frames [default False]', default=0)
parser.add_argument('-nframe', help='select number of frames [default all]', default=0)
args = parser.parse_args()

# =============================================================================
# Load data
# =============================================================================

def load(fname, n_frames=None, resize = 200):
    frames, fps = import_video(fname+'.mp4')
    if resize: frames = resize_max(frames,resize)
    frames = preprocess(frames)
    frames = frames[:n_frames,...] if n_frames is not None else frames
    return frames, fps

nframe = int(args.nframe) if int(args.nframe)!=0 else None 
frames, fps = load(args.tar, n_frames=nframe, resize=int(args.resize))
print('\n\n\nLoaded video %s has shape %s\n\n\n'%(args.tar, str(frames.shape)))


# =============================================================================
# Set parameters
# =============================================================================
GPU = int(args.gpu)
octave_scale = 1.5
loss_w = {'ss':1, 'ts':1}

config = dict(

        
    hyperparams = dict(device='/GPU:'+str(GPU), equalize_losses=1e4,
                     batch = 1, pad=5, blend=1, col_trans= ['ss']),
              
    opt_info = dict(verbose=200, verbose_plot=False),
              
    out_settings = dict(fname= 'stst', fps=fps),
              
    opt_settings=[
        
        dict(iters=2000, optimizer=dict(params={'learning_rate': 0.001}), regularizer_w={'tv': 0.1},
             scale=dict(octave=-2, octave_scale=octave_scale), loss_w=loss_w), 
        
        dict(iters=1000, optimizer=dict(params={'learning_rate': 0.002}), regularizer_w={'tv': 0.1},
             scale=dict(octave=-1, octave_scale=octave_scale), loss_w=loss_w), 

        dict(iters=500, optimizer=dict(params={'learning_rate': 0.005}), regularizer_w={'tv': 0.1},
             scale=dict(octave=0, octave_scale=octave_scale), loss_w=loss_w)
    ]
              
)



# =============================================================================
# Run optimization process
# =============================================================================
targets = {'ss': frames, 'ts': frames}
    

stst = SpaceTimeStyleTransfer(targets)
gen = stst.generate(config)
stst.save_results()
