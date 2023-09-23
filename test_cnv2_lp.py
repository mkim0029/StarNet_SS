import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'cnv2_utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import str2bool, parseArguments, LARS
from network import build_encoder, load_model_state
from analysis_fns import (plot_progress, plot_val_MAEs, encoder_predict,
                          predict_labels,
                          plot_resid_violinplot,
                          plot_resid, tsne_comparison, compare_veracity)

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
source_val_survey = config['DATA']['source_val_survey']
if source_val_survey.lower()=='none':
    source_val_survey = None
target_val_survey = config['DATA']['target_val_survey']
if target_val_survey.lower()=='none':
    target_val_survey = None
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
batch_size = int(config['MAE TRAINING']['batch_size'])
mask_ratio = float(config['MAE TRAINING']['mask_ratio'])

wave_grid = np.load(wave_grid_file)

# Build network
model = build_encoder(config, device, model_name)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'_lp.pth.tar')
model, losses, _ = load_model_state(model, model_filename)

# Create data loaders
source_train_dataset = SpectraDataset(source_data_file, 
                                      dataset='train', 
                                      label_keys=label_keys,
                                      label_survey=source_val_survey,
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
                                      label_survey=source_val_survey,
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
              y_lims=[(0,3), (0,0.15)],lp=True,
             savename=os.path.join(figs_dir, '%s_lp_train_progress.png'%model_name))

plot_val_MAEs(losses, label_keys, 
              y_lims=[(0.,600.), (0.,0.6), (0.,1.0), (0.,.3), (0,0.2)],
             savename=os.path.join(figs_dir, '%s_val_progress.png'%model_name))

# Predict on source
(label_keys, tgt_labels, pred_labels, feature_maps_src) = predict_labels(model,
                                                                source_val_dataloader, 
                                                                device=device)
# Save predictions
np.save(os.path.join(results_dir, '%s_source_preds.npy'%model_name), pred_labels)
np.save(os.path.join(results_dir, '%s_source_tgts.npy'%model_name), tgt_labels)
np.save(os.path.join(results_dir, '%s_source_feature_maps.npy'%model_name), feature_maps_src)

# Save a plot
plot_resid_violinplot(label_keys, tgt_labels, pred_labels,
                      y_lims=[1000, 1.2, 1.5, 0.8], 
                      savename=os.path.join(figs_dir, '%s_source_val_results.png'%model_name))

(label_keys, tgt_labels, pred_labels, feature_maps_tgt) = predict_labels(model,
                                                                target_val_dataloader, 
                                                                device=device)
'''
(tgt_mm_labels2, tgt_um_labels, 
 pred_mm_labels2, pred_um_labels, feature_maps_tgt2) = predict_labels(model, target_train_dataloader, 
                                                  device=device, take_mode=False)

pred_mm_labels = np.vstack((pred_mm_labels, pred_mm_labels2))
tgt_mm_labels = np.vstack((tgt_mm_labels, tgt_mm_labels2))
'''
# Save predictions
np.save(os.path.join(results_dir, '%s_target_preds.npy'%model_name), pred_labels)
np.save(os.path.join(results_dir, '%s_target_tgts.npy'%model_name), tgt_labels)
np.save(os.path.join(results_dir, '%s_target_feature_maps.npy'%model_name), feature_maps_tgt)

# Save a plot
if len(np.unique(tgt_labels[:,0]))<40:
    plot_resid_violinplot(label_keys, tgt_labels, pred_labels,
                          y_lims=[1000, 1.2, 1.5, 0.8], 
                          savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))
else:
    plot_resid(label_keys, tgt_labels, pred_labels,
               y_lims = [1000, 1.2, 1.5, 0.8], 
               savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))


# Plot isochrone comparison
print('Plotting veracity...')
isochrone_fn = os.path.join(cur_dir, 'data/isochrone_data.h5')
compare_veracity(isochrone_fn, teff1=tgt_labels[:,0], 
                logg1=tgt_labels[:,2], 
                feh1=tgt_labels[:,1], 
                teff2=pred_labels[:,0], 
                logg2=pred_labels[:,2], 
                feh2=pred_labels[:,1],
                label1='APOGEE', label2='StarNet',
                feh_min=-1, feh_max=0.5, 
                feh_lines=[-1., -0.5, 0.0, 0.5], 
                savename=os.path.join(figs_dir, '%s_target_val_veracity.png'%model_name))

print('Finished.')

tsne_comparison(feature_maps_src, feature_maps_tgt, 
                label1=r'$\mathbf{\mathcal{Z}_{synth}}$',
                label2=r'$\mathbf{\mathcal{Z}_{obs}}$',
               savename=os.path.join(figs_dir, '%s_lp_feature_tsne.png'%model_name))
