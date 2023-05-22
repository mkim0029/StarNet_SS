import os
# Directory of training script
cur_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_dir,'utils'))
from data_loader import SpectraDataset, batch_to_device
from training_utils import (parseArguments,CosineSimilarityLoss, run_iter, 
                            str2bool, val_iter)
from network import StarNet, build_starnet, load_model_state

import configparser
import time
import numpy as np
import h5py
import torch
from collections import defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

# np.random.seed(1)
# torch.manual_seed(1)

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
multimodal_keys = eval(config['DATA']['multimodal_keys'])
unimodal_keys = eval(config['DATA']['unimodal_keys'])
continuum_normalize = str2bool(config['DATA']['continuum_normalize'])
divide_by_median = str2bool(config['DATA']['divide_by_median'])
apply_dropout = str2bool(config['DATA']['apply_dropout'])
add_noise_to_source = str2bool(config['DATA']['add_noise_to_source'])
random_chunk = str2bool(config['DATA']['random_chunk'])
overlap = float(config['DATA']['overlap'])
max_noise_factor = float(config['DATA']['max_noise_factor'])
channel_indices = eval(config['DATA']['channel_indices'])
std_min = float(config['DATA']['std_min'])
batch_size = int(config['TRAINING']['batchsize'])
chunk_size = int(config['TRAINING']['chunk_size'])
lr = float(config['TRAINING']['lr'])
final_lr_factor = float(config['TRAINING']['final_lr_factor'])
weight_decay = float(config['TRAINING']['weight_decay'])
total_batch_iters = float(config['TRAINING']['total_batch_iters'])
source_mm_weights = torch.tensor(eval(config['TRAINING']['source_mm_weights'])).to(device)
source_um_weights = torch.tensor(eval(config['TRAINING']['source_um_weights'])).to(device)
source_feature_weight = float(config['TRAINING']['source_feature_weight'])
target_feature_weight = float(config['TRAINING']['target_feature_weight'])
target_task_weights = torch.tensor(eval(config['TRAINING']['target_task_weights'])).to(device)
source_task_weights = torch.tensor(eval(config['TRAINING']['source_task_weights'])).to(device)
feat_loss_fn = config['TRAINING']['feat_loss_fn']

# Calculate multimodal values from source training set
with h5py.File(source_data_file, "r") as f:
    mutlimodal_vals = []
    for k in multimodal_keys:
        vals = np.unique(f[k + ' train'][:]).astype(np.float32)
        mutlimodal_vals.append(torch.from_numpy(vals).to(device))

# Build network
model = build_starnet(config, device, model_name, mutlimodal_vals)

# Construct optimizer
optimizer = torch.optim.AdamW(model.all_parameters(), 
                             lr,
                             weight_decay=weight_decay, 
                             betas=(0.9, 0.999))

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=int(total_batch_iters), 
                                                   pct_start=0.05, anneal_strategy='cos', 
                                                   cycle_momentum=True, base_momentum=0.85, 
                                                   max_momentum=0.95, div_factor=25.0, 
                                                   final_div_factor=final_lr_factor, three_phase=False)

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')
model, losses, cur_iter, chunk_indices, chunk_weights = load_model_state(model,
                                                                         model_filename, 
                                                                         optimizer, 
                                                                         lr_scheduler)

# Multi GPUs
model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
if num_gpus>1:
    batch_size *= num_gpus

# Create data loaders
source_train_dataset = SpectraDataset(source_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                           continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median,
                                           chunk_size=chunk_size, 
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=std_min,
                                           apply_dropout=apply_dropout,
                                           add_noise=add_noise_to_source,
                                           max_noise_factor=max_noise_factor,
                                           random_chunk=random_chunk,
                                           overlap=overlap,
                                           channel_indices=channel_indices)

source_train_dataloader = torch.utils.data.DataLoader(source_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=3,
                                                      pin_memory=True)

source_val_dataset = SpectraDataset(source_data_file, 
                                                  dataset='val', 
                                                  wave_grid_file=wave_grid_file, 
                                                  multimodal_keys=multimodal_keys,
                                                  unimodal_keys=unimodal_keys,
                                                  continuum_normalize=continuum_normalize,
                                                  divide_by_median=divide_by_median,
                                                  chunk_size=chunk_size,  
                                                  tasks=model.module.tasks, 
                                                  task_means=model.module.task_means.cpu().numpy(), 
                                                  task_stds=model.module.task_stds.cpu().numpy(),
                                                  median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                  overlap=overlap,
                                                  channel_indices=channel_indices)

source_val_dataloader = torch.utils.data.DataLoader(source_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

target_train_dataset = SpectraDataset(target_data_file, 
                                           dataset='train', 
                                           wave_grid_file=wave_grid_file, 
                                           multimodal_keys=multimodal_keys,
                                           unimodal_keys=unimodal_keys,
                                           continuum_normalize=continuum_normalize,
                                           divide_by_median=divide_by_median, 
                                           chunk_size=chunk_size, 
                                           tasks=model.module.tasks, 
                                           task_means=model.module.task_means.cpu().numpy(), 
                                           task_stds=model.module.task_stds.cpu().numpy(),
                                           median_thresh=0., std_min=std_min, 
                                           apply_dropout=False,
                                           random_chunk=random_chunk,
                                           overlap=overlap,
                                          channel_indices=channel_indices)

target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=3,
                                                      pin_memory=True)

target_val_dataset = SpectraDataset(target_data_file, 
                                                  dataset='val', 
                                                  wave_grid_file=wave_grid_file, 
                                                  multimodal_keys=multimodal_keys,
                                                  unimodal_keys=unimodal_keys,
                                                  continuum_normalize=continuum_normalize,
                                                  divide_by_median=divide_by_median, 
                                                  chunk_size=chunk_size, 
                                                  tasks=model.module.tasks, 
                                                  task_means=model.module.task_means.cpu().numpy(), 
                                                  task_stds=model.module.task_stds.cpu().numpy(),
                                                  median_thresh=0., std_min=std_min, 
                                                  random_chunk=random_chunk,
                                                   overlap=overlap,
                                                 channel_indices=channel_indices)

target_val_dataloader = torch.utils.data.DataLoader(target_val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=3,
                                                    pin_memory=True)

print('The source training set consists of %i spectra.' % (len(source_train_dataset)))
print('The source validation set consists of %i spectra.' % (len(source_val_dataset)))

print('The target training set consists of %i spectra.' % (len(target_train_dataset)))
print('The target validation set consists of %i spectra.' % (len(target_val_dataset)))

if 'mse' in feat_loss_fn.lower():
    feat_loss_fn = torch.nn.MSELoss()
elif 'l1' in feat_loss_fn.lower():
    feat_loss_fn = torch.nn.L1Loss()
elif 'cosine' in feat_loss_fn.lower():
    feat_loss_fn = CosineSimilarityLoss()

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
                                                                 source_mm_weights,
                                                                 source_um_weights,
                                                                 source_feature_weight,
                                                                 target_feature_weight,
                                                                 source_task_weights,
                                                                 target_task_weights,
                                                                 feat_loss_fn,
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
                        losses_cp = val_iter(model, 
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
                if model.module.num_mm_labels>0:
                    print('\t\tStellar Multimodal Loss: %0.3f' % (losses['train_src_mm_labels'][-1]))
                    print('\t\tChunk Multimodal Loss: %0.3f' % (losses['train_src_mm_labels_chunk'][-1]))
                if model.module.num_um_labels>0:
                    print('\t\tStellar Unimodal Loss: %0.3f' % (losses['train_src_um_labels'][-1]))
                    print('\t\tChunk Unimodal Loss: %0.3f' % (losses['train_src_um_labels_chunk'][-1]))
                if len(model.module.tasks)>0:
                    for i, task in enumerate(model.module.tasks):
                        print('\t\tSource %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_src_tasks'][-1][i]))
                        print('\t\tTarget %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_tgt_tasks'][-1][i]))
                        print('\t\tSource Chunk %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_src_tasks_chunk'][-1][i]))
                        print('\t\tTarget Chunk %s Task Loss: %0.3f' % (task.capitalize(),
                                                                  losses['train_tgt_tasks_chunk'][-1][i]))
                if model.module.use_split_convs:
                    print('\t\tSource Feature Loss: %0.3f' % (losses['train_src_feats'][-1]))
                    print('\t\tTarget Feature Loss: %0.3f' % (losses['train_tgt_feats'][-1]))
                print('\tValidation Dataset')
                #print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                if model.module.num_mm_labels>0:
                    for i, key in enumerate(model.module.multimodal_keys):
                        print('\t\tSource %s MAE: %0.3f' % (key.capitalize(),
                                                            losses['val_src_'+key][-1]))
                        print('\t\tTarget %s MAE: %0.3f' % (key.capitalize(),
                                                            losses['val_tgt_'+key][-1]))
                if model.module.num_um_labels>0:
                    for i, key in enumerate(model.module.unimodal_keys):
                        print('\t\tSource %s MAE: %0.3f' % (key.capitalize(),
                                                            losses['val_src_'+key][-1]))
                        print('\t\tTarget %s MAE: %0.3f' % (key.capitalize(),
                                                            losses['val_tgt_'+key][-1]))

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
                            'model' : model.module.state_dict(),
                            'classifier models': [net.state_dict() for net in model.module.label_classifiers]},
                            model_filename)

                cp_start_time = time.time()

            if cur_iter>(total_batch_iters):
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.module.state_dict(),
                            'classifier models': [net.state_dict() for net in model.module.label_classifiers]},
                            model_filename)
                # Finish training
                break 

# Run the training
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, cur_iter)
    print('\nTraining complete.')
