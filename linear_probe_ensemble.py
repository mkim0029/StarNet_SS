import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
import glob
sys.path.append(os.path.join(cur_dir,'mae_utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import str2bool, parseArguments, LARS
from mae_network import build_mae, load_model_state
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
config.read(config_dir+model_name+'_a.ini')

# Training parameters from config file
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])
multimodal_keys = eval(config['DATA']['multimodal_keys'])
unimodal_keys = eval(config['DATA']['unimodal_keys'])
target_val_survey = config['DATA']['target_val_survey']
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
batch_size = int(config['TRAINING']['batch_size'])
mask_ratio = float(config['TRAINING']['mask_ratio'])

wave_grid = np.load(wave_grid_file)


# Calculate multimodal values from source training set
with h5py.File(source_data_file, "r") as f:
    mutlimodal_vals = []
    for k in multimodal_keys:
        vals = np.unique(f[k + ' train'][:]).astype(np.float32)
        mutlimodal_vals.append(torch.from_numpy(vals).to(device))

# Build network
model = build_mae(config, device, model_name, mutlimodal_vals)

# Load ensemble of models
model_filenames = glob.glob(os.path.join(model_dir, '%s_*_lp.pth.tar'%(model_name)))

models = []
for model_filename in model_filenames:
    print(model_filename)
    model, losses, _ ,_ = load_model_state(model, model_filename)
    
    model.eval_mode()
    models.append(model)

# Create data loaders
source_train_dataset = SpectraDataset(source_data_file, 
                                      dataset='train', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=3,
                                                      pin_memory=True)

source_val_dataset = SpectraDataset(source_data_file, 
                                      dataset='val', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

target_train_dataset = SpectraDataset(target_data_file, 
                                      dataset='train', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
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
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
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

pred_labels = np.zeros((len(target_train_dataset)+len(target_val_dataset), 
                       len(multimodal_keys) + len(unimodal_keys)))
for model in models:
    label_keys, tgt_labels1, pred_labels1, _ = predict_labels(model,
                                                              target_train_dataloader, 
                                                                    device=device, 
                                                                    take_mode=False)

    _, tgt_labels2, pred_labels2, _ = predict_labels(model,
                                                                    target_val_dataloader, 
                                                                    device=device, 
                                                                    take_mode=False)
    tgt_labels1 = np.vstack((tgt_labels1, tgt_labels2))
    pred_labels1 = np.vstack((pred_labels1, pred_labels2))
    
    pred_labels += 1/len(models) * pred_labels1

# Save predictions
np.save(os.path.join(results_dir, '%s_target_preds.npy'%model_name), pred_labels)
np.save(os.path.join(results_dir, '%s_target_tgts.npy'%model_name), tgt_labels1)


# Plot isochrone comparison
print('Plotting veracity...')
isochrone_fn = os.path.join(cur_dir, 'data/isochrone_data.h5')
compare_veracity(isochrone_fn, teff1=tgt_labels1[:,0], 
                logg1=tgt_labels1[:,2], 
                feh1=tgt_labels1[:,1], 
                teff2=pred_labels[:,0], 
                logg2=pred_labels[:,2], 
                feh2=pred_labels[:,1],
                label1='APOGEE', label2='StarNet',
                feh_min=-1, feh_max=0.5, 
                feh_lines=[-1., -0.5, 0.0, 0.5], 
                savename=os.path.join(figs_dir, '%s_target_val_veracity_ensemble.png'%model_name))
