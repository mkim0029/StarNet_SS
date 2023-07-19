import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'starnet_utils'))
from data_loader import SpectraDatasetSimple, batch_to_device
from network import build_starnet, load_model_state
from training_utils import parseArguments, run_iter

import configparser
import numpy as np
import h5py
from collections import defaultdict
import time

import torch

# Determine if we're using GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# np.random.seed(1)
# torch.manual_seed(1)

print('Using Torch version: %s' % (torch.__version__))
print('Using a %s device.' % (device))

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir

# Directories
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')

# Training parameters from config file
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
label_keys = eval(config['DATA']['label_keys'])
max_noise_factor = float(config['DATA']['max_noise_factor'])
batch_size = int(config['TRAINING']['batch_size'])
lr = float(config['TRAINING']['lr'])
final_lr_factor = float(config['TRAINING']['final_lr_factor'])
weight_decay = float(config['TRAINING']['weight_decay'])
total_batch_iters = int(config['TRAINING']['total_batch_iters'])

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

# Construct optimizer
'''optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=lr, 
                              weight_decay=weight_decay, 
                              betas=(0.9, 0.999))'''
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=lr,
                             weight_decay=weight_decay,
                             betas=(0.9, 0.999))

print(optimizer)
    
# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                                   total_steps=int(total_batch_iters), 
                                                   pct_start=0.05, anneal_strategy='linear', 
                                                   cycle_momentum=True, 
                                                   base_momentum=0.85, 
                                                   max_momentum=0.95, div_factor=25.0, 
                                                   final_div_factor=final_lr_factor, 
                                                   three_phase=False)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename,
                                           optimizer=optimizer, 
                                           lr_scheduler=lr_scheduler)

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

def train_network(model, optimizer, lr_scheduler, cur_iter):
    print('Training the network with a batch size of %i ...' % (batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while cur_iter < (total_batch_iters):
        # Iterate through training dataset
        for train_batch in source_train_dataloader:
        
            # Switch to GPU if available
            train_batch = batch_to_device(train_batch, device)
            
            # Run iteration on a batch of training samples            
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, optimizer, 
                                                                 lr_scheduler, train_batch, 
                                                                 losses_cp, 
                                                                 mode='train', 
                                                                 dataset='source')
            
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:
                with torch.no_grad():
                    for dataset, dataloader in zip(['source', 'target'],
                                                   [source_val_dataloader, 
                                                    target_val_dataloader]):
                        for val_batch in dataloader:
                            # Switch to GPU if available
                            val_batch = batch_to_device(val_batch, device)

                            # Run evaluation on a batch of validation samples 
                            model, _, _, losses_cp = run_iter(model, 
                                                              optimizer, 
                                                              lr_scheduler, 
                                                              val_batch, 
                                                              losses_cp, 
                                                              mode='val', 
                                                              dataset=dataset)

                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tLoss: %0.3f'% (losses['train_source_loss'][-1]))
                
                print('\tValidation Dataset')
                print('\t\tSource Loss: %0.3f'% (losses['val_source_loss'][-1]))
                print('\t\tTarget Loss: %0.3f'% (losses['val_target_loss'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)

            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time*60:
                
                # Save periodically
                print('Saving network...')
                torch.save({'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.state_dict(), 
                            'batch_iters' : cur_iter,
                            'losses' : losses},
                           model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.state_dict(), 
                            'batch_iters' : cur_iter,
                            'losses' : losses},
                           model_filename)
                
                # Finish training
                break 
                
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, cur_iter)
    print('\nTraining complete.')