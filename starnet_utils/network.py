import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import os
from training_utils import str2bool
from collections import defaultdict

def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    
    f = mod.forward(torch.autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]

class StarNet(nn.Module):
    def __init__(self, num_pixels, num_filters, filter_length, 
                 pool_length, num_hidden, num_labels,
                 spectra_mean, spectra_std, labels_mean, labels_std, device):
        super().__init__()
        
        # Save distribution of training data
        self.spectra_mean = spectra_mean
        self.spectra_std = spectra_std
        self.labels_mean = torch.tensor(np.asarray(labels_mean).astype(np.float32)).to(device)
        self.labels_std = torch.tensor(np.asarray(labels_std).astype(np.float32)).to(device)
        
        # Convolutional and pooling layers
        self.conv1 = nn.Conv1d(1, num_filters[0], filter_length)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], filter_length)
        self.pool = nn.MaxPool1d(pool_length, pool_length)
        
        # Determine shape after pooling
        pool_output_shape = compute_out_size((1,num_pixels), 
                                             nn.Sequential(self.conv1, 
                                                           self.conv2, 
                                                           self.pool))
        
        # Fully connected layers
        self.fc1 = nn.Linear(pool_output_shape[0]*pool_output_shape[1], num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.output = nn.Linear(num_hidden[1], num_labels)

    def normalize(self, data, data_mean, data_std):
        '''Normalize inputs to have zero-mean and unit-variance.'''
        return (data - data_mean) / data_std
    
    def denormalize(self, data, data_mean, data_std):
        '''Undo the normalization to put the data back in the original scale.'''
        return data * data_std + data_mean
        
    def forward(self, x, norm_in=True, denorm_out=False):
        
        if norm_in:
            # Normalize spectra to have zero-mean and unit variance
            x = self.normalize(x, self.spectra_mean, self.spectra_std)
        
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        if denorm_out:
            # Denormalize predictions to be on the original scale of the labels
            x = self.denormalize(x, self.labels_mean, self.labels_std)
            
        return x

def build_starnet(config, device, model_name, 
                  spectra_mean, spectra_std, 
                  labels_mean, labels_std):
    
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
    print('\nBuilding network...')

    # Construct architecture based on config file
    model = StarNet(num_pixels=int(config['ARCHITECTURE']['spectrum_size']), 
                    num_filters=eval(config['ARCHITECTURE']['num_filters']), 
                    filter_length=int(config['ARCHITECTURE']['filter_length']),  
                    pool_length=int(config['ARCHITECTURE']['pool_length']), 
                    num_hidden=eval(config['ARCHITECTURE']['num_hidden']), 
                    num_labels=len(eval(config['DATA']['label_keys'])),
                    spectra_mean=spectra_mean, spectra_std=spectra_std, 
                    labels_mean=labels_mean, labels_std=labels_std,
                   device=device)
    
    # Switch to GPU if available
    model = model.to(device)

    # Print summary
    summary(model, (int(config['ARCHITECTURE']['spectrum_size']), ))
        
    return model

def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None):
    
    # Check for pre-trained weights
    if os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model weights...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)

        # Dictionary of loss values
        losses = defaultdict(list, dict(checkpoint['losses']))

        # Current batch iteration of training
        cur_iter = checkpoint['batch_iters']+1

        # Load optimizer states
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter