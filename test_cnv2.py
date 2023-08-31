import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'cnv2_utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import str2bool, parseArguments, LARS
from network import build_mae, load_model_state
from analysis_fns import (plot_progress, mae_predict, encoder_predict, 
                          tsne_comparison, plot_5_samples, plot_spec_resid_density)

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

# Training parameters from config file
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])
label_keys = eval(config['DATA']['label_keys'])
target_val_survey = config['DATA']['target_val_survey']
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
use_prev_ae = str2bool(config['MAE TRAINING']['use_prev_ae'])
prev_ae_name = config['MAE TRAINING']['prev_ae_name']
batch_size = int(config['MAE TRAINING']['batch_size'])
mask_ratio = float(config['MAE TRAINING']['mask_ratio'])

wave_grid = np.load(wave_grid_file)

# Build network
model = build_mae(config, device, model_name)

# Load model state from previous training (if any)
if use_prev_ae:
    model_filename = os.path.join(model_dir, prev_ae_name+'.pth.tar')
else:
    model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename)

# Create data loaders
source_train_dataset = SpectraDataset(source_data_file, 
                                      dataset='train', 
                                      label_keys=label_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=3,
                                                      pin_memory=True)

source_val_dataset = SpectraDataset(source_data_file, 
                                      dataset='val', 
                                      label_keys=label_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

target_train_dataset = SpectraDataset(target_data_file, 
                                      dataset='train', 
                                      label_keys=label_keys,
                                      label_survey=target_val_survey,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=3,
                                                      pin_memory=True)

target_val_dataset = SpectraDataset(target_data_file, 
                                      dataset='val', 
                                      label_keys=label_keys,
                                      label_survey=target_val_survey,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))


# Plot the training progress
plot_progress(losses, 
              y_lims=[(0,0.1), (0.12,0.22)],
             savename=os.path.join(figs_dir, '%s_train_progress.png'%model_name))

# Predict on masked spectra
(pred_spectra_src, mask_spectra_src, 
 orig_spectra_src, latents_src) = mae_predict(model, 
                                              source_val_dataloader, 
                                              device, mask_ratio=mask_ratio)

(pred_spectra_tgt, mask_spectra_tgt, 
 orig_spectra_tgt, latents_tgt) = mae_predict(model, 
                                              target_val_dataloader, 
                                              device, mask_ratio=mask_ratio)

# Encode full spectra
latents_full_src = encoder_predict(model, source_val_dataloader, 
                                   device, mask_ratio=0.)

latents_full_tgt = encoder_predict(model, target_val_dataloader, 
                                   device, mask_ratio=0.)

# Save predictions
np.save(os.path.join(results_dir, '%s_source_feature_maps.npy'%model_name), latents_full_src)
np.save(os.path.join(results_dir, '%s_target_feature_maps.npy'%model_name), latents_tgt)

plot_5_samples(wave_grid, orig_spectra_src, mask_spectra_src, pred_spectra_src, 
               savename=os.path.join(figs_dir, '%s_src_samples.png'%model_name))

plot_5_samples(wave_grid, orig_spectra_tgt, mask_spectra_tgt, pred_spectra_tgt, 
               savename=os.path.join(figs_dir, '%s_tgt_samples.png'%model_name))

plot_spec_resid_density(wave_grid, orig_spectra_src, mask_spectra_src, pred_spectra_src, 
                            ylim=(-0.05, 0.05), hist=True, kde=True,
                            dist_bins=180, hex_grid=100, bias='med', scatter='std',
                            bias_label=r'$\overline{{m}}$ \ ',
                            scatter_label=r'$s$ \ ',
                            cmap="ocean_r", 
                        savename=os.path.join(figs_dir, '%s_spec_resid_src.png'%model_name))

plot_spec_resid_density(wave_grid, orig_spectra_tgt, mask_spectra_tgt, pred_spectra_tgt, 
                            ylim=(-0.15, 0.15), hist=True, kde=True,
                            dist_bins=180, hex_grid=100, bias='med', scatter='std',
                            bias_label=r'$\overline{{m}}$ \ ',
                            scatter_label=r'$s$ \ ',
                            cmap="ocean_r", 
                        savename=os.path.join(figs_dir, '%s_spec_resid_tgt.png'%model_name))

tsne_comparison(latents_full_src, latents_full_tgt, 
                label1=r'$\mathbf{\mathcal{Z}_{synth}}$',
                label2=r'$\mathbf{\mathcal{Z}_{obs}}$',
               savename=os.path.join(figs_dir, '%s_feature_tsne.png'%model_name))
