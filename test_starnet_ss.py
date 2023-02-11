import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'utils'))
from data_loader import WeaveSpectraDataset, WeaveSpectraDatasetInference, batch_to_device
from training_utils import (parseArguments, str2bool)
from network import StarNet, build_starnet, load_model_state
from analysis_fns import (plot_progress, plot_val_MAEs, predict_labels, 
                          predict_ensemble, plot_resid, plot_resid_boxplot,
                           plot_one_to_one, plot_wave_sigma)

import configparser
import time
import numpy as np
import h5py
import torch
from collections import defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

print('Using Torch version: %s' % (torch.__version__))
print('Using a %s device with %i gpus' % (device, num_gpus))

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters =args.verbose_iters
cp_time = args.cp_time
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
add_noise_to_source = str2bool(config['DATA']['add_noise_to_source'])
random_chunk = str2bool(config['DATA']['random_chunk'])
overlap = float(config['DATA']['overlap'])
channel_indices = eval(config['DATA']['channel_indices'])
std_min = float(config['DATA']['std_min'])
batch_size = int(config['TRAINING']['batchsize'])

# Calculate multimodal values from source training set
with h5py.File(source_data_file, "r") as f:
    mutlimodal_vals = []
    for k in multimodal_keys:
        vals = np.unique(f[k + ' train'][:]).astype(np.float32)
        mutlimodal_vals.append(torch.from_numpy(vals).to(device))

# Build network
model = build_starnet(config, device, model_name, mutlimodal_vals)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename)

# Multi GPUs
model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
if num_gpus>1:
    batch_size *= num_gpus
    
load_second_chunk = model.module.use_split_convs
# Create dataset for loading spectra
source_train_dataset = WeaveSpectraDatasetInference(source_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                           continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median,
                                           num_fluxes=model.module.num_fluxes, 
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap,
                                                  channel_indices=channel_indices)

source_val_dataset = WeaveSpectraDatasetInference(source_data_file, 
                                         dataset='val', 
                                         wave_grid_file=wave_grid_file, 
                                         multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                         continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median,
                                         num_fluxes=model.module.num_fluxes, 
                                         tasks=model.module.tasks, 
                                         task_means=model.module.task_means.cpu().numpy(), 
                                         task_stds=model.module.task_stds.cpu().numpy(),
                                         median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap,
                                                  channel_indices=channel_indices)

target_train_dataset = WeaveSpectraDatasetInference(target_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                           continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median, 
                                           num_fluxes=model.module.num_fluxes,
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap,
                                                  channel_indices=channel_indices)

target_val_dataset = WeaveSpectraDatasetInference(target_data_file, 
                                         dataset='val', 
                                         wave_grid_file=wave_grid_file, 
                                         multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                         continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median,
                                         num_fluxes=model.module.num_fluxes,
                                         tasks=model.module.tasks, 
                                         task_means=model.module.task_means.cpu().numpy(), 
                                         task_stds=model.module.task_stds.cpu().numpy(),
                                         median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap,
                                                  channel_indices=channel_indices)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

plot_progress(losses, model.module.tasks, 
              y_lims=[(1,3),(0.,2),(0.0,0.7),(0,0.07),(0,0.5),(0,0.05),(0,0.3),(0,0.3),(0,0.2)],
             savename=os.path.join(figs_dir, '%s_train_progress.png'%model_name))

plot_val_MAEs(losses, multimodal_keys+unimodal_keys, 
              y_lims=[(0.,200.), (0.,0.1), (0.,0.25), (0.,.1), (0.,0.2), (0.,30)],
             savename=os.path.join(figs_dir, '%s_val_progress.png'%model_name))


(tgt_stellar_labels, pred_stellar_labels, 
sigma_stellar_labels) = predict_labels(model, source_train_dataset, device=device, take_mode=True)

plot_resid_boxplot(multimodal_keys, tgt_stellar_labels, pred_stellar_labels,
                   y_lims=[300, 0.6, 1, 0.2],
                   savename=os.path.join(figs_dir, '%s_source_val_results.png'%model_name))

(tgt_stellar_labels, pred_stellar_labels, 
sigma_stellar_labels) = predict_labels(model, target_train_dataset, device=device, take_mode=True)

plot_resid_boxplot(multimodal_keys, tgt_stellar_labels, pred_stellar_labels,
                   y_lims=[300, 0.6, 1, 0.2],
                   savename=os.path.join(figs_dir, '%s_target_val_results.png'%model_name))
