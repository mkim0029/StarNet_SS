import torch
from torch import nn
import torch.nn.functional as F

import argparse
import numpy as np

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
                        type=int, default=1000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=5)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args
        
def run_iter(model, optimizer, lr_scheduler, batch, 
             losses_cp, mode, dataset):

    if mode=='train':
        model.train()
    else:
        model.eval()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward propagation
    label_preds = model(batch['spectrum'], 
                        norm_in=True, 
                        denorm_out=False)

    # Compute mean-squared-error loss between predictions and normalized targets
    loss = torch.nn.MSELoss()(label_preds, 
                              model.normalize(batch['labels'], 
                                              model.labels_mean,
                                              model.labels_std))
    
    if mode=='train':

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
    
        # Adjust learning rate
        lr_scheduler.step()

        # Save loss
        losses_cp['train_%s_loss'%dataset].append(float(loss))

    else:
        
        losses_cp['val_%s_loss'%dataset].append(float(loss))
    
    return model, optimizer, lr_scheduler, losses_cp