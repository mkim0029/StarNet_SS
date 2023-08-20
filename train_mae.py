import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'mae_utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import parseArguments, mae_iter, str2bool
from mae_network import build_mae, load_model_state

import configparser
import time
import numpy as np
import h5py
from collections import defaultdict

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

# np.random.seed(1)
# torch.manual_seed(1)

print('Using Torch version: %s' % (torch.__version__))
print('Using a %s device with %i gpus' % (device, num_gpus))

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir

# Directories
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')

# Training parameters from config file
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])
multimodal_keys = eval(config['DATA']['multimodal_keys'])
unimodal_keys = eval(config['DATA']['unimodal_keys'])
target_val_survey = config['DATA']['target_val_survey']
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
apply_dropout = str2bool(config['DATA']['apply_dropout'])
add_noise_to_source = str2bool(config['DATA']['add_noise_to_source'])
max_noise_factor = float(config['DATA']['max_noise_factor'])
std_min = float(config['DATA']['std_min'])
augs = eval(config['DATA']['augs'])
aug_means = eval(config['DATA']['aug_means'])
aug_stds = eval(config['DATA']['aug_stds'])
use_prev_ae = str2bool(config['TRAINING']['use_prev_ae'])
prev_ae_name = config['TRAINING']['prev_ae_name']
batch_size = int(config['TRAINING']['batch_size'])
lr = float(config['TRAINING']['lr'])
final_lr_factor = float(config['TRAINING']['final_lr_factor'])
target_loss_weight = float(config['TRAINING']['target_loss_weight'])
weight_decay = float(config['TRAINING']['weight_decay'])
total_batch_iters = int(config['TRAINING']['total_batch_iters'])
mask_ratio = float(config['TRAINING']['mask_ratio'])
        
# Calculate multimodal values from source training set
with h5py.File(source_data_file, "r") as f:
    mutlimodal_vals = []
    for k in multimodal_keys:
        vals = np.unique(f[k + ' train'][:]).astype(np.float32)
        mutlimodal_vals.append(torch.from_numpy(vals).to(device))
        
# Build network
model = build_mae(config, device, model_name, mutlimodal_vals)

# Construct optimizer
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=lr, 
                              weight_decay=weight_decay,
                              betas=(0.9, 0.95))
print(optimizer)


# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                                   total_steps=int(total_batch_iters), 
                                                   pct_start=0.05, anneal_strategy='cos', 
                                                   cycle_momentum=True, 
                                                   base_momentum=0.85, 
                                                   max_momentum=0.95, div_factor=25.0, 
                                                   final_div_factor=final_lr_factor, 
                                                   three_phase=False)

# Load model state from previous training (if any)
if use_prev_ae:
    model_filename = os.path.join(model_dir, prev_ae_name+'.pth.tar')
else:
    model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter, _ = load_model_state(model, model_filename, 
                                           optimizer, lr_scheduler)#, loss_scaler)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
    
# Create data loaders
source_train_dataset = SpectraDataset(source_data_file, 
                                      dataset='train', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      median_thresh=0., std_min=std_min,
                                     add_noise=add_noise_to_source, max_noise_factor=max_noise_factor)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=5,
                                                      pin_memory=True)

source_val_dataset = SpectraDataset(source_data_file, 
                                      dataset='val', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      median_thresh=0., std_min=std_min)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=5,
                                                    pin_memory=True)

target_train_dataset = SpectraDataset(target_data_file, 
                                      dataset='train', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      label_survey=target_val_survey,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      median_thresh=0., std_min=std_min)

target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=5,
                                                      pin_memory=True)

target_val_dataset = SpectraDataset(target_data_file, 
                                      dataset='val', 
                                      multimodal_keys=multimodal_keys,
                                      unimodal_keys=unimodal_keys,
                                      label_survey=target_val_survey,
                                      continuum_normalize=continuum_normalize,
                                      divide_by_median=divide_by_median,
                                      median_thresh=0., std_min=std_min)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=5,
                                                    pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

def train_network(model, optimizer, lr_scheduler, cur_iter):
    print('Training the network with a batch size of %i ...' % (batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while cur_iter < (total_batch_iters):
        # Iterate through both training datasets simultaneously
        source_dataloader_iterator = iter(source_train_dataloader)
        for target_train_batch in target_train_dataloader:

            try:
                source_train_batch = next(source_dataloader_iterator)
            except StopIteration:
                source_dataloader_iterator = iter(source_train_dataloader)
                source_train_batch = next(source_dataloader_iterator)
        
            # Switch to GPU if available
            source_train_batch = batch_to_device(source_train_batch, device)
            target_train_batch = batch_to_device(target_train_batch, device)
            
            # Run iteration on a batch of training samples            
            model, optimizer, lr_scheduler, losses_cp = mae_iter(model, 
                                                                 optimizer, 
                                                                 lr_scheduler, 
                                                                 #loss_scaler, 
                                                                 source_train_batch, 
                                                                 target_train_batch, 
                                                                 mask_ratio,
                                                                 target_loss_weight,
                                                                 losses_cp, 
                                                                 cur_iter, 
                                                                 total_batch_iters, 
                                                                 mode='train',
                                                                device=device)
            
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:
                with torch.no_grad():
                    for source_val_batch, target_val_batch in zip(source_val_dataloader,
                                                                  target_val_dataloader):
                        # Switch to GPU if available
                        source_val_batch = batch_to_device(source_val_batch, device)
                        target_val_batch = batch_to_device(target_val_batch, device)

                        # Run evaluation on a batch of validation samples 
                        model, optimizer, lr_scheduler, losses_cp = mae_iter(model, 
                                                                 optimizer, 
                                                                 lr_scheduler, 
                                                                 #loss_scaler, 
                                                                 source_val_batch, 
                                                                 target_val_batch, 
                                                                 mask_ratio,
                                                                 target_loss_weight,
                                                                 losses_cp, 
                                                                 cur_iter, 
                                                                 total_batch_iters, 
                                                                 mode='val',
                                                                            device=device)

                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                print('\t\tSource Loss: %0.3f' % (losses['train_src_loss'][-1]))
                print('\t\tTarget Loss: %0.3f' % (losses['train_tgt_loss'][-1]))
                
                print('\tValidation Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                print('\t\tSource Loss: %0.3f' % (losses['val_src_loss'][-1]))
                print('\t\tTarget Loss: %0.3f' % (losses['val_tgt_loss'][-1]))
                print('\t\tFeature Matching: %0.3f' % (losses['val_feats'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)


            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time*60:
                
                # Save periodically
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                #'loss_scaler': loss_scaler.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.state_dict(),
                            'classifier models': [net.state_dict() for net in model.label_classifiers]},
                                model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                #'loss_scaler': loss_scaler.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.state_dict(),
                            'classifier models': [net.state_dict() for net in model.label_classifiers]},
                                model_filename)
                # Finish training
                break 
                
# Run the training
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, cur_iter)
    print('\nTraining complete.')