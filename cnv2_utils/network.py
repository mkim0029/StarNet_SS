import torch
from torch import nn
import torch.nn.functional as F

import os
from training_utils import str2bool
from itertools import chain
from collections import defaultdict
import math

import torch
import torch.nn as nn

from convnextv2 import FCMAE1D, ConvNeXtV2, load_sparse_to_nonsparse

def build_mae(config, device, model_name):
    
    # Display model configuration
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))

    # Construct Network
    print('\nBuilding networks...')
    model = FCMAE1D(img_size=int(config['MAE ARCHITECTURE']['spectrum_size']),
                    in_chans=1,
                    depths=eval(config['MAE ARCHITECTURE']['encoder_depths']),
                    dims=eval(config['MAE ARCHITECTURE']['encoder_dims']),
                    decoder_depth=int(config['MAE ARCHITECTURE']['decoder_depth']),
                    decoder_embed_dim=int(config['MAE ARCHITECTURE']['decoder_dim']),
                    patch_size=int(config['MAE ARCHITECTURE']['patch_size']),
                    input_mean=float(config['DATA']['spectra_mean']),
                    input_std=float(config['DATA']['spectra_std']))
    
    model.to(device)

    # Display model architectures
    print('MAE Architecture:')
    print(model)
        
    return model

def build_encoder(config, device, model_name):
    
    # Display model configuration
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))

    # Construct Network
    print('\nBuilding networks...')
    model = ConvNeXtV2(in_chans=1, 
                       depths=eval(config['MAE ARCHITECTURE']['encoder_depths']), 
                       dims=eval(config['MAE ARCHITECTURE']['encoder_dims']), 
                       stem_width=4, 
                       input_mean=float(config['DATA']['spectra_mean']),
                       input_std=float(config['DATA']['spectra_std']), 
                       label_keys=eval(config['DATA']['label_keys']), 
                       label_means=eval(config['DATA']['label_means']), 
                       label_stds=eval(config['DATA']['label_stds']), 
                       dropout=float(config['LINEAR PROBE TRAINING']['dropout']),
                       lp_enc_layers=int(config['LINEAR PROBE TRAINING']['num_enc_layers']),
                      device=device)
    
    model.to(device)

    # Display model architectures
    print('Encoder Architecture:')
    print(model)
        
    return model

def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None, sparse_to_nonsparse=False):
    
    if sparse_to_nonsparse:
        print('\nLoading pretrained MAE weights to finetune...')
        model = load_sparse_to_nonsparse(model, model_filename)
        losses = defaultdict(list)
        cur_iter = 1
    
    # Check for pre-trained weights
    elif os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model weights...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)
        losses = defaultdict(list, dict(checkpoint['losses']))
        cur_iter = checkpoint['batch_iters']+1

        # Load optimizer states
        try:
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except ValueError:
            pass
        
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter
