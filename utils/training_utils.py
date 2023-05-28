import torch
import argparse
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

import sys
import os
cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir,'utils'))
from data_loader import batch_to_device

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def loss_fn(y_true, y_pred, y_sigma):
    '''For a model that outputs the mean and std of each label.'''
    return torch.mean(torch.log(y_sigma)/2+ (y_true - y_pred)**2/(2*y_sigma)) + 5

def task_loss_fn(y_true, y_pred):
    '''Take average of each task loss separately.'''
    return torch.mean((y_true - y_pred)**2, axis=0)

def CosineSimilarityLoss(eps=1e-6):
    cos = torch.nn.CosineSimilarity(dim=1, eps=eps)
    def loss(inp, tgt):
        return torch.mean(1 - cos(inp, tgt))
    return loss

def run_iter(model, src_batch, tgt_batch, optimizer, lr_scheduler, 
             source_mm_weights, source_um_weights, source_feature_weight,
             target_feature_weight, source_task_weights, target_task_weights, 
             feat_loss_fn, losses_cp, mode='train'):
        
    if mode=='train':
        model.module.train_mode()
    else:
        model.module.eval_mode()
        
    total_loss = 0.
    
    # Compute prediction on source batch.
    # First on the entire spectra and then on chunks from the spectra.
    model_outputs_src = model(src_batch['spectrum'],
                              src_batch['spectrum index'],
                              norm_in=True, denorm_out=False, return_feats=True)
    model_outputs_src_chunk = model(src_batch['spectrum chunk'],
                                    src_batch['chunk index'],
                                    norm_in=True, denorm_out=False, return_feats=True)
    
    # Compute prediction on target batch
    # First on the entire spectra and then on chunks from the spectra.
    model_outputs_tgt = model(tgt_batch['spectrum'],
                              tgt_batch['spectrum index'],
                              norm_in=True, denorm_out=False, return_feats=True)
    model_outputs_tgt_chunk = model(tgt_batch['spectrum chunk'],
                                    tgt_batch['chunk index'],
                                    norm_in=True, denorm_out=False, return_feats=True)
        
    if model.module.num_mm_labels>0:
        # Compute the average loss on stellar class labels
        src_mm_loss_tot = 0.
        src_mm_loss_tot_chunk = 0.
        
        # Convert target label values to classes
        src_classes = model.module.multimodal_to_class(src_batch['multimodal labels'])
        for i in range(model.module.num_mm_labels):
            # Evaluate loss on predictions from the entire spectra
            src_mm_loss = torch.nn.NLLLoss()(model_outputs_src['multimodal labels'][i], 
                                             src_classes[i])
            src_mm_loss_tot += 1/model.module.num_mm_labels * src_mm_loss
            
            # Evaluate loss on predictions from the spectrum chunks
            src_mm_loss_chunk = torch.nn.NLLLoss()(model_outputs_src_chunk['multimodal labels'][i], 
                                                   src_classes[i])
            src_mm_loss_tot_chunk += 1/model.module.num_mm_labels * src_mm_loss_chunk
            
            # Add to total loss
            if source_mm_weights[i]>0:
                total_loss = total_loss + source_mm_weights[i] * src_mm_loss/2
                total_loss = total_loss + source_mm_weights[i] * src_mm_loss_chunk/2
    else:
        src_mm_loss_tot = 0.
        src_mm_loss_tot_chunk = 0.
        
    if model.module.num_um_labels>0:
        src_um_loss_tot = 0.
        src_um_loss_tot_chunk = 0.
        # Normalize target labels
        src_um_labels = model.module.normalize_unimodal(src_batch['unimodal labels'])
        for i in range(model.module.num_um_labels):
            # Evaluate loss on predictions from the entire spectra
            src_um_loss = torch.nn.MSELoss()(model_outputs_src['unimodal labels'][:,i], 
                                             src_um_labels[:,i])
            src_um_loss_tot += 1/model.module.num_um_labels * src_um_loss
            
            # Evaluate loss on predictions from the spectrum chunks
            src_um_loss_chunk = torch.nn.MSELoss()(model_outputs_src_chunk['unimodal labels'][:,i], 
                                             src_um_labels[:,i])
            src_um_loss_tot_chunk += 1/model.module.num_um_labels * src_um_loss_chunk
            
            # Add to total loss
            if source_um_weights[i]>0:
                total_loss = total_loss + source_um_weights[i] * src_um_loss/2
                total_loss = total_loss + source_um_weights[i] * src_um_loss_chunk/2
        
    else:
        src_um_loss_tot = 0.
        src_um_loss_tot_chunk = 0.
        
    # Compare feature maps of chunk vs full spectrum.
    # Do this for both the source and target domains.
    src_feature_loss = feat_loss_fn(model_outputs_src['feature map'], 
                                    model_outputs_src_chunk['feature map'])
        
    tgt_feature_loss = feat_loss_fn(model_outputs_tgt['feature map'], 
                                    model_outputs_tgt_chunk['feature map'])
       
    # Add to total loss
    if source_feature_weight>0:
        total_loss = total_loss + source_feature_weight*src_feature_loss
    if target_feature_weight>0:
        total_loss = total_loss + target_feature_weight*tgt_feature_loss
        
    if len(model.module.tasks)>0:
        # Compute loss on task labels
        src_task_losses = task_loss_fn(model.module.normalize_tasks(src_batch['task labels full']), 
                                       model_outputs_src['task labels'])
        src_task_losses_chunk = task_loss_fn(model.module.normalize_tasks(src_batch['task labels chunk']), 
                                             model_outputs_src_chunk['task labels'])
        tgt_task_losses = task_loss_fn(model.module.normalize_tasks(tgt_batch['task labels full']), 
                                       model_outputs_tgt['task labels'])
        tgt_task_losses_chunk = task_loss_fn(model.module.normalize_tasks(tgt_batch['task labels chunk']), 
                                       model_outputs_tgt_chunk['task labels'])
        # Add to total loss
        for i in range(len(src_task_losses)):
            if source_task_weights[i]>0:
                total_loss = total_loss + 1/len(src_task_losses)*src_task_losses[i]/2*source_task_weights[i]
                total_loss = total_loss + 1/len(src_task_losses)*src_task_losses_chunk[i]/2*source_task_weights[i]
            if target_task_weights[i]>0:
                total_loss = total_loss + 1/len(tgt_task_losses)*tgt_task_losses[i]/2*target_task_weights[i]
                total_loss = total_loss + 1/len(tgt_task_losses)*tgt_task_losses_chunk[i]/2*target_task_weights[i]
        
    if mode=='train':        
        # Update the gradients
        total_loss.backward()

        # Save loss and metrics
        losses_cp['train_loss'].append(float(total_loss))
        losses_cp['train_src_feats'].append(float(src_feature_loss))
        losses_cp['train_tgt_feats'].append(float(tgt_feature_loss))
        if model.module.num_mm_labels>0:
            losses_cp['train_src_mm_labels'].append(float(src_mm_loss_tot))
            losses_cp['train_src_mm_labels_chunk'].append(float(src_mm_loss_tot_chunk))
        if model.module.num_um_labels>0:
            losses_cp['train_src_um_labels'].append(float(src_um_loss_tot))
            losses_cp['train_src_um_labels_chunk'].append(float(src_um_loss_tot_chunk))
        if len(model.module.tasks)>0:
            losses_cp['train_src_tasks'].append(src_task_losses.cpu().data.numpy().tolist())
            losses_cp['train_src_tasks_chunk'].append(src_task_losses_chunk.cpu().data.numpy().tolist())
            losses_cp['train_tgt_tasks'].append(tgt_task_losses.cpu().data.numpy().tolist())
            losses_cp['train_tgt_tasks_chunk'].append(tgt_task_losses_chunk.cpu().data.numpy().tolist())

        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        # Adjust learning rate
        lr_scheduler.step()

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(total_loss))
        losses_cp['val_src_feats'].append(float(src_feature_loss))
        losses_cp['val_tgt_feats'].append(float(tgt_feature_loss))
        if model.module.num_mm_labels>0:
            losses_cp['val_src_mm_labels'].append(float(src_mm_loss_tot))
            losses_cp['val_src_mm_labels_chunk'].append(float(src_mm_loss_tot_chunk))
        if model.module.num_um_labels>0:
            losses_cp['val_src_um_labels'].append(float(src_um_loss_tot))
            losses_cp['val_src_um_labels_chunk'].append(float(src_um_loss_tot))
        if len(model.module.tasks)>0:
            losses_cp['val_src_tasks'].append(src_task_losses.cpu().data.numpy().tolist())
            losses_cp['val_src_tasks_chunk'].append(src_task_losses.cpu().data.numpy().tolist())
            losses_cp['val_tgt_tasks'].append(tgt_task_losses.cpu().data.numpy().tolist())
            losses_cp['val_tgt_tasks_chunk'].append(tgt_task_losses.cpu().data.numpy().tolist())
                
    return model, optimizer, lr_scheduler, losses_cp

def val_iter(model, src_batch, tgt_batch, losses_cp):
        
    model.module.eval_mode()
        
    # Compute prediction on source batch
    model_outputs_src = model(src_batch['spectrum'],
                              src_batch['spectrum index'],
                              norm_in=True, denorm_out=True, return_feats=True)
    # Compute prediction on target batch
    model_outputs_tgt = model(tgt_batch['spectrum'],
                              tgt_batch['spectrum index'],
                              norm_in=True, denorm_out=True, return_feats=True)
        
    # Compute Mean Abs Error on multimodal label predictions
    src_mm_losses = []
    tgt_mm_losses = []
    for i in range(model.module.num_mm_labels):
        src_mm_losses.append(torch.nn.L1Loss()(model_outputs_src['multimodal labels'][:,i], 
                                               src_batch['multimodal labels'][:,i]))
        tgt_mm_losses.append(torch.nn.L1Loss()(model_outputs_tgt['multimodal labels'][:,i], 
                                               tgt_batch['multimodal labels'][:,i]))
    
    # Compute mean absolute error on unimodal label predictions
    src_um_losses = []
    tgt_um_losses = []
    for i in range(model.module.num_um_labels):
        src_um_losses.append(torch.nn.L1Loss()(model_outputs_src['unimodal labels'][:,i], 
                                               src_batch['unimodal labels'][:,i]))
        tgt_um_losses.append(torch.nn.L1Loss()(model_outputs_tgt['unimodal labels'][:,i], 
                                               tgt_batch['unimodal labels'][:,i]))
        
    
    # Evaluate distance between features
    
    # Compute max and min of each feature
    max_feat = torch.max(torch.cat((model_outputs_src['feature map'], 
                                    model_outputs_tgt['feature map']), 0), 
                         dim=0).values
    min_feat = torch.min(torch.cat((model_outputs_src['feature map'], 
                                    model_outputs_tgt['feature map']), 0), 
                         dim=0).values

    # Normalize each feature between 0 and 1 across the entire batch
    model_feats_src_norm = ( (model_outputs_src['feature map'] - min_feat) /
                             (max_feat-min_feat+1e-8) )
    model_feats_tgt_norm = ( (model_outputs_tgt['feature map'] - min_feat) /
                             (max_feat-min_feat+1e-8) )
    
    # Compute mean absolute error
    feat_loss = torch.mean(torch.abs(model_feats_src_norm-model_feats_tgt_norm))    
        
    # Save losses    
    for src_val, tgt_val, label_key in zip(src_mm_losses, tgt_mm_losses, model.module.multimodal_keys):
        losses_cp['val_src_'+label_key].append(float(src_val))
        losses_cp['val_tgt_'+label_key].append(float(tgt_val))
    for src_val, tgt_val, label_key in zip(src_um_losses, tgt_um_losses, model.module.unimodal_keys):
        losses_cp['val_src_'+label_key].append(float(src_val))
        losses_cp['val_tgt_'+label_key].append(float(tgt_val))
        
    losses_cp['val_feats'].append(float(feat_loss))
                
    return losses_cp