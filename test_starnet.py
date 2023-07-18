import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'starnet_utils'))
from data_loader import SpectraDatasetSimple, batch_to_device
from training_utils import (parseArguments, str2bool)
from network import build_starnet, load_model_state
from analysis_fns import (plot_progress, dataset_inference, plot_resid_violinplot, plot_resid, compare_veracity)

import configparser
import time
import numpy as np
import h5py
import torch
from collections import defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using Torch version: %s' % (torch.__version__))

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
data_dir = args.data_dir

# Directories
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
figs_dir = os.path.join(cur_dir, 'figures/')
results_dir = os.path.join(cur_dir, 'results/')
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')

source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
label_keys = eval(config['DATA']['label_keys'])
max_noise_factor = float(config['DATA']['max_noise_factor'])
batch_size = int(config['TRAINING']['batch_size'])

# Collect mean and std of the training data
with h5py.File(source_data_file, "r") as f:
    labels_mean = [np.mean(f[k + ' train'][:]) for k in label_keys]
    labels_std = [np.std(f[k + ' train'][:]) for k in label_keys]
    spectra_mean = np.mean(f['spectra train'][:]) 
    spectra_std = np.mean(f['spectra train'][:])

# Build network
model = build_starnet(config, device, model_name, 
                      spectra_mean, spectra_std, 
                      labels_mean, labels_std)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename)

# Create data loaders
source_train_dataset = SpectraDatasetSimple(source_data_file, 
                                           dataset='train', 
                                           label_keys=label_keys,
                                           max_noise_factor=max_noise_factor)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=4,
                                                      pin_memory=True)

source_val_dataset = SpectraDatasetSimple(source_data_file, 
                                          dataset='val', 
                                          label_keys=label_keys,
                                          max_noise_factor=max_noise_factor)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   pin_memory=True)

target_val_dataset = SpectraDatasetSimple(target_data_file, 
                                          dataset='val', 
                                          label_keys=label_keys,
                                          max_noise_factor=0.0)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

# Plot the training progress
print('Plotting progress...')
plot_progress(losses, 
              y_lims=[(0,0.3)],
             savename=os.path.join(figs_dir, '%s_train_progress.png'%model_name))

# Predict on source validation data
print('Evaluating source spectra...')
tgt_labels, pred_labels = dataset_inference(model, source_val_dataloader, device)

# Save a plot
plot_resid_violinplot(label_keys, tgt_labels, pred_labels,
                      y_lims=[1000, 1.2, 1.5, 0.8], 
                      savename=os.path.join(figs_dir, '%s_source_val_results.png'%model_name))

# Predict on target validation data
print('Evaluating target spectra...')
tgt_labels, pred_labels = dataset_inference(model, target_val_dataloader, device)


plot_resid(label_keys, tgt_labels, pred_labels,
               y_lims = [1000, 1.2, 1.5, 0.8], 
               savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))
    
isochrone_fn = 'data/isochrone_data.h5'

# Plot isochrone comparison
print('Plotting veracity...')
compare_veracity(isochrone_fn, teff1=tgt_labels[:,0], 
                logg1=tgt_labels[:,2], 
                feh1=tgt_labels[:,1], 
                teff2=pred_labels[:,0], 
                logg2=pred_labels[:,2], 
                feh2=pred_labels[:,1],
                label1='GAIA', label2='StarNet',
                feh_min=-1, feh_max=0.5, 
                feh_lines=[-1., -0.5, 0.0, 0.5], 
                savename=os.path.join(figs_dir, '%s_target_val_veracity.png'%model_name))

print('Finished.')