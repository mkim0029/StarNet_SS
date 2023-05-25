import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import (parseArguments, str2bool)
from network import StarNet, build_starnet, load_model_state
from analysis_fns import (plot_progress, plot_val_MAEs, predict_labels, 
                          predict_ensemble, plot_resid, plot_resid_violinplot,
                           plot_one_to_one, plot_wave_sigma, plot_resid)

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

# TRAINING PARAMETERS
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])
multimodal_keys = eval(config['DATA']['multimodal_keys'])
unimodal_keys = eval(config['DATA']['unimodal_keys'])
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
batch_size = int(config['TRAINING']['batchsize'])

# Calculate multimodal values from source training set
with h5py.File(source_data_file, "r") as f:
    mutlimodal_vals = []
    for k in multimodal_keys:
        vals = np.unique(f[k + ' train'][:]).astype(np.float32)
        mutlimodal_vals.append(torch.from_numpy(vals).to(device))

# Build network
model = build_starnet(config, device, model_name, mutlimodal_vals)

# Load model state from previous training
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename)

# Create dataset for loading spectra
source_train_dataset = SpectraDataset(source_data_file, 
                                      dataset='train', 
                                      wave_grid_file=wave_grid_file, 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      inference_mode=True)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=5,
                                                      pin_memory=True)

source_val_dataset = SpectraDataset(source_data_file, 
                                    dataset='val', 
                                    wave_grid_file=wave_grid_file, 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      inference_mode=True)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=5,
                                                      pin_memory=True)

target_train_dataset = SpectraDataset(target_data_file, 
                                      dataset='train', 
                                      wave_grid_file=wave_grid_file, 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      inference_mode=True)

target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=5,
                                                      pin_memory=True)

target_val_dataset = SpectraDataset(target_data_file, 
                                    dataset='val', 
                                    wave_grid_file=wave_grid_file, 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      inference_mode=True)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=5,
                                                      pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

def predict_labels(model, dataloader, device, batchsize=16, take_mode=False):
    
    print('Predicting on %i batches...' % (len(dataloader)))
    try:
        model.eval_mode()
    except AttributeError:
        model.module.eval_mode()
        

    tgt_mm_labels = []
    tgt_um_labels = []
    pred_mm_labels = []
    pred_um_labels = []
    feature_maps = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for batch in dataloader:
            
            batch = batch_to_device(batch, device)

            # Collect target data
            tgt_mm_labels.append(batch['multimodal labels'].data.cpu().numpy())
            if len(batch['unimodal labels'][0])>0:
                tgt_um_labels.append(batch['unimodal labels'].data.cpu().numpy())

            # Perform forward propagation
            try:
                model_outputs = model(batch['spectrum'],
                                  batch['spectrum index'],
                                  norm_in=True, denorm_out=True, return_feats=True)
            except AttributeError:
                model_outputs = model.module(batch['spectrum'],
                                  batch['spectrum index'],
                                  norm_in=True, denorm_out=True, return_feats=True)

                
            # Save predictions
            pred_mm_labels.append(model_outputs['multimodal labels'].data.cpu().numpy())
            feature_maps.append(model_outputs['feature map'].data.cpu().numpy())
            if len(batch['unimodal labels'][0])>0:
                pred_um_labels.append(model_outputs['unimodal labels'].data.cpu().numpy())

        tgt_mm_labels = np.vstack(tgt_mm_labels)
        pred_mm_labels = np.vstack(pred_mm_labels)
        feature_maps = np.vstack(feature_maps)
        if len(tgt_um_labels)>0:
            tgt_um_labels = np.vstack(tgt_um_labels)
            pred_um_labels = np.vstack(pred_um_labels)
        
    return tgt_mm_labels, tgt_um_labels, pred_mm_labels, pred_um_labels, feature_maps

# Plot the training progress
plot_progress(losses, model.tasks, 
              y_lims=[(0,17),(0.,4),(0.0,0.1),(0,0.01),(0,1.),
                      (0,0.5),(0,0.1),(0,0.4),(0,1.1),(0,0.6),(0,0.6),(0,0.1),(0,0.6)],
             savename=os.path.join(figs_dir, '%s_train_progress.png'%model_name))

plot_val_MAEs(losses, multimodal_keys+unimodal_keys, 
              y_lims=[(0.,600.), (0.,0.6), (0.,1.0), (0.,.3), (0,40)],
             savename=os.path.join(figs_dir, '%s_val_progress.png'%model_name))

# Predict on source
(tgt_mm_labels, tgt_um_labels, 
 pred_mm_labels, pred_um_labels, feature_maps) = predict_labels(model, source_val_dataloader, 
                                                  device=device, take_mode=False)
# Save predictions
np.save(os.path.join(results_dir, '%s_source_mm_preds.npy'%model_name), pred_mm_labels)
np.save(os.path.join(results_dir, '%s_source_mm_tgts.npy'%model_name), tgt_mm_labels)

# Save a plot
plot_resid_violinplot(multimodal_keys, tgt_mm_labels, pred_mm_labels,
                      y_lims=[1000, 1.2, 1.5, 0.8], 
                      savename=os.path.join(figs_dir, '%s_source_val_results.png'%model_name))
'''
# Predict on target
(tgt_mm_labels, tgt_um_labels, 
 pred_mm_labels, pred_um_labels) = predict_labels(model, target_train_dataset, 
                                                  device=device, take_mode=False, 
                                                  combine_batch_probs=True,
                                                 chunk_indices=torch.tensor(chunk_indices),
                                                chunk_weights=torch.tensor(chunk_weights))
'''
(tgt_mm_labels, tgt_um_labels, 
 pred_mm_labels, pred_um_labels, feature_maps) = predict_labels(model, target_val_dataloader, 
                                                  device=device, take_mode=False)
'''
pred_mm_labels = np.vstack((pred_mm_labels, pred_mm_labels2))
tgt_mm_labels = np.vstack((tgt_mm_labels, tgt_mm_labels2))
'''
# Save predictions
np.save(os.path.join(results_dir, '%s_target_mm_preds.npy'%model_name), pred_mm_labels)
np.save(os.path.join(results_dir, '%s_target_mm_tgts.npy'%model_name), tgt_mm_labels)
np.save(os.path.join(results_dir, '%s_feature_maps.npy'%model_name), feature_maps)

# Save a plot
if len(np.unique(tgt_mm_labels[:,0]))<40:
    plot_resid_violinplot(multimodal_keys, tgt_mm_labels, pred_mm_labels,
                          y_lims=[1000, 1.2, 1.5, 0.8], 
                          savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))
else:
    plot_resid(multimodal_keys, tgt_mm_labels, pred_mm_labels,
               y_lims = [1000, 1.2, 1.5, 0.8], 
               savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))
