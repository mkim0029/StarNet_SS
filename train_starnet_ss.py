import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'utils'))
from data_loader import WeaveSpectraDataset, WeaveSpectraDatasetInference, batch_to_device
from training_utils import (parseArguments, WarmupLinearSchedule, run_iter, 
                            str2bool, compare_val_sample)
from network import StarNet, build_starnet, load_model_state

import configparser
import time
import numpy as np
import h5py
import torch
from collections import defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

#np.random.seed(1)
#torch.manual_seed(1)

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
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')
    
# TRAINING PARAMETERS
source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])
label_keys = eval(config['DATA']['label_keys'])
normalize_spectra = str2bool(config['DATA']['normalize_spectra'])
split_channels = str2bool(config['DATA']['split_channels'])
add_noise_to_source = str2bool(config['DATA']['add_noise_to_source'])
random_chunk = str2bool(config['DATA']['random_chunk'])
overlap = float(config['DATA']['overlap'])
batch_size = int(config['TRAINING']['batchsize'])
lr_warmup = int(config['TRAINING']['lr_warmup'])
min_lr = float(config['TRAINING']['min_lr'])
min_lr_iters = int(config['TRAINING']['min_lr_iters'])
total_batch_iters = float(config['TRAINING']['total_batch_iters'])
target_task_weights = torch.tensor(eval(config['TRAINING']['target_task_weights'])).to(device)
source_task_weights = torch.tensor(eval(config['TRAINING']['source_task_weights'])).to(device)

# Build network
model = build_starnet(config, device, model_name)

# Construct optimizer
optimizer = torch.optim.Adam(model.all_parameters(), 
                             0., weight_decay=0., 
                             betas=(0.9, 0.999))

# Learning rate scheduler
lr_scheduler = WarmupLinearSchedule(optimizer, 
                                    warmup_steps=lr_warmup,
                                    min_lr=min_lr,
                                    t_total=min_lr_iters)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter = load_model_state(model, model_filename, optimizer, lr_scheduler)

# Multi GPUs
model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
if num_gpus>1:
    batch_size *= num_gpus

# Create data loaders
source_train_dataset = WeaveSpectraDataset(source_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           label_keys=label_keys,
                                           normalize_spectra=normalize_spectra,
                                           split_channels=split_channels,
                                           num_fluxes=model.module.num_fluxes, 
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=0.01,
                                           apply_dropout=True,
                                           add_noise=add_noise_to_source,
                                           random_chunk=random_chunk,
                                           overlap=overlap)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=3,
                                                      pin_memory=True)

source_val_dataset = WeaveSpectraDatasetInference(source_data_file, 
                                                  dataset='val', 
                                                  wave_grid_file=wave_grid_file, 
                                                  label_keys=label_keys,
                                                  normalize_spectra=normalize_spectra,
                                                  split_channels=split_channels, 
                                                  num_fluxes=model.module.num_fluxes, 
                                                  tasks=model.module.tasks, 
                                                  task_means=model.module.task_means.cpu().numpy(), 
                                                  task_stds=model.module.task_stds.cpu().numpy(),
                                                  median_thresh=0., std_min=0.01, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset, 
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

target_train_dataset = WeaveSpectraDataset(target_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           label_keys=label_keys,
                                           normalize_spectra=normalize_spectra, 
                                           split_channels=split_channels,
                                           num_fluxes=model.module.num_fluxes,
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=0.01, 
                                           apply_dropout=False,
                                           random_chunk=random_chunk,
                                           overlap=overlap)

target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=3,
                                                      pin_memory=True)

target_val_dataset = WeaveSpectraDatasetInference(target_data_file, 
                                                  dataset='val', 
                                                  wave_grid_file=wave_grid_file, 
                                                  label_keys=label_keys,
                                                  normalize_spectra=normalize_spectra, 
                                                  split_channels=split_channels,
                                                  num_fluxes=model.module.num_fluxes,
                                                  tasks=model.module.tasks, 
                                                  task_means=model.module.task_means.cpu().numpy(), 
                                                  task_stds=model.module.task_stds.cpu().numpy(),
                                                  median_thresh=0., std_min=0.01, 
                                                  random_chunk=random_chunk,
                                                   overlap=overlap)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset, 
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

def train_network(model, optimizer, lr_scheduler, cur_iter):
    print('Training the network with a batch size of %i...' % (batch_size))
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
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, 
                                                                 source_train_batch,
                                                                 target_train_batch,
                                                                 optimizer, 
                                                                 lr_scheduler,
                                                                 target_task_weights,
                                                                 source_task_weights,
                                                                 losses_cp, 
                                                                 mode='train')

            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:
                with torch.no_grad():
                    for source_val_batch, target_val_batch in zip(source_val_dataloader,
                                                                  target_val_dataloader):
                        # Switch to GPU if available
                        source_val_batch = batch_to_device(source_val_batch, device)
                        target_val_batch = batch_to_device(target_val_batch, device)

                        # Run evaluation on a batch of validation samples 
                        losses_cp = compare_val_sample(model, 
                                                       source_val_batch, 
                                                       target_val_batch, 
                                                       losses_cp)

                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)

                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                print('\t\tStellar Label Loss: %0.3f' % (losses['train_src_labels'][-1]))
                if len(model.module.tasks)>0:
                    for i, task in enumerate(model.module.tasks):
                        print('\t\tSource %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_src_tasks'][-1][i]))
                        print('\t\tTarget %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_tgt_tasks'][-1][i]))
                print('\tValidation Dataset')
                #print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                print('\t\tSource Label Loss: %0.3f' % (losses['val_src_labels'][-1]))
                print('\t\tTarget Label Loss: %0.3f' % (losses['val_tgt_labels'][-1]))
                print('\t\tFeature Map Score: %0.3f' % (losses['val_feats'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)


            # Increase the iteration
            cur_iter += 1

            if time.time() - cp_start_time >= cp_time*60:
                # Save periodically
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.module.state_dict()},
                            model_filename)

                cp_start_time = time.time()

            if cur_iter>(total_batch_iters):
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.module.state_dict()},
                           model_filename)
                # Finish training
                break                

# Run the training
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, cur_iter)
    print('\nTraining complete.')