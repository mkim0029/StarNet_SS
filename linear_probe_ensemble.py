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
                          plot_resid, tsne_comparison, compare_veracity, plot_resid_hexbin)

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
#if data_dir is None:
data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
if os.path.exists(config_dir+model_name+'_a.ini'):
    config.read(config_dir+model_name+'_a.ini')
    ensemble = True
else:
    config.read(config_dir+model_name+'.ini')
    ensemble = False

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
        
# Load survey labels and spectra
surveys = ['APOGEE', 'GAIA']
survey_labels = []
with h5py.File(target_data_file, "r") as f:
    # Load spectrum
    obs_spectra = np.concatenate((f['spectra train'][:], f['spectra val'][:]))
    obs_spectra[obs_spectra<-1] = -1.
    
    for survey in surveys:
        labels = []
        for key in (multimodal_keys+unimodal_keys):
            labels.append(np.concatenate((f['%s %s train' % (survey, key)][:],
                                          f['%s %s val' % (survey, key)][:])))
        labels = np.vstack(labels).T
        survey_labels.append(labels)
survey_labels = np.array(survey_labels)

# Remove samples with nan labels
indices = np.where(~np.isnan(np.hstack((survey_labels[0], survey_labels[1]))).any(axis=1))[0]
obs_spectra = torch.from_numpy(np.asarray(obs_spectra[indices]).astype(np.float32)).to(device)
survey_labels = survey_labels[:,indices]

# Build network
model = build_mae(config, device, model_name, mutlimodal_vals)

# Load ensemble of models
if ensemble:
    model_filenames = glob.glob(os.path.join(model_dir, '%s_*_lp.pth.tar'%(model_name)))
    model_filenames = np.sort(model_filenames)
else:
    model_filenames = [os.path.join(model_dir, '%s_lp.pth.tar'%(model_name))]

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



def predict_ensemble_labels(models, spectra, batchsize=2048):
    
    print('Predicting on %i batches...' % (len(spectra)/batchsize))
        
    pred_avg_labels = []
    pred_std_labels = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for i in range(0, len(spectra), batchsize):
            
            ensemble_preds = []
            for model in models:
                # Perform forward propagation
                try:
                    # Compute prediction on source batch
                    model_outputs = model.forward_labels(spectra[i:i+batchsize], 
                                                         norm_in=True, denorm_out=True, 
                                                         return_feats=False)
                except AttributeError:
                    model_outputs = model.module.forward_labels(spectra[i:i+batchsize], 
                                                                norm_in=True, denorm_out=True, 
                                                                return_feats=False)


                # Save predictions
                if (model.num_mm_labels>0)&(model.num_um_labels>0):
                    ensemble_preds.append(np.hstack((model_outputs['multimodal labels'].data.cpu().numpy(), 
                                                     model_outputs['unimodal labels'].data.cpu().numpy())))
                elif model.num_mm_labels>0:
                    ensemble_preds.append(model_outputs['multimodal labels'].data.cpu().numpy())
                elif model.num_um_labels>0:
                    ensemble_preds.append(model_outputs['unimodal labels'].data.cpu().numpy())
            
            # Take average
            pred_avg_labels.append(np.mean(np.array(ensemble_preds), axis=0))
            pred_std_labels.append(np.std(np.array(ensemble_preds), axis=0))
            

    pred_avg_labels = np.concatenate(pred_avg_labels)
    pred_std_labels = np.concatenate(pred_std_labels)
    label_keys = model.multimodal_keys + model.unimodal_keys
        
    return label_keys, pred_avg_labels, pred_std_labels

label_keys, pred_avg_labels, pred_std_labels = predict_ensemble_labels(models, obs_spectra, batchsize=2048)

# Save predictions
np.save(os.path.join(results_dir, '%s_target_avg_preds.npy'%model_name), pred_avg_labels)
np.save(os.path.join(results_dir, '%s_target_std_preds.npy'%model_name), pred_std_labels)

print('Plotting veracity...')
isochrone_fn = os.path.join(cur_dir, 'data/isochrone_data.h5')
compare_veracity(isochrone_fn, teff1=survey_labels[0,:,0], 
                logg1=survey_labels[0,:,2], 
                feh1=survey_labels[0,:,1], 
                teff2=pred_avg_labels[:,0], 
                logg2=pred_avg_labels[:,2], 
                feh2=pred_avg_labels[:,1],
                label1='APOGEE', label2='StarNet-MAE',
                feh_min=-1, feh_max=0.5, 
                feh_lines=[-1., -0.5, 0.0, 0.5], 
                savename=os.path.join(figs_dir, '%s_target_veracity_ensemble.png'%model_name))

plot_resid_hexbin(unimodal_keys, survey_labels[0], pred_avg_labels,
                  y_lims = [1000, 1.2, 1.5, 0.8], x_label='APOGEE',
                 gridsize=(100,15), max_counts=100, cmap='ocean_r', 
                  n_std=3, 
                savename=os.path.join(figs_dir, '%s_target_resid_density.png'%model_name))
