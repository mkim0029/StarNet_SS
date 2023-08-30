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

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
        
def latent_distance(latent1, latent2):
    # Compute max and min of each feature
    max_feat = torch.max(torch.cat((latent1, 
                                    latent2), 0), 
                         dim=0).values
    min_feat = torch.min(torch.cat((latent1, 
                                    latent2), 0), 
                         dim=0).values

    # Normalize each feature between 0 and 1 across the entire batch
    latent1_norm = ( (latent1 - min_feat) /
                    (max_feat-min_feat+1e-8) )
    latent2_norm = ( (latent2 - min_feat) /
                    (max_feat-min_feat+1e-8) )
    
    # Compute mean absolute error
    return torch.mean(torch.abs(latent1_norm-latent2_norm))   
        
def mae_iter(model, optimizer, lr_scheduler, #loss_scaler, 
               src_batch, tgt_batch, mask_ratio, target_loss_weight,
               losses_cp, cur_iter, total_batch_iters, mode):

    if mode=='train':
        model.train(True)
    else:
        model.eval()

    # Zero the gradients
    optimizer.zero_grad()

    '''
    mr1 = 0.1
    mr2 = 0.95
    if mode=='train':
        mask_ratio = (mr2 - mr1) * torch.rand(1) + mr1
    else:
        mask_ratio = mr1
    '''
    
    # Compute predictions and losses
    #with torch.cuda.amp.autocast():
    #loss, _, _, latent = model(torch.concatenate((src_batch['spectrum'],
    #                                                  tgt_batch['spectrum']), dim=0), 
    #                                   mask_ratio=mask_ratio, norm_in=True)
    src_loss, _, _, src_latent = model(src_batch['spectrum'], 
                                       mask_ratio=mask_ratio, norm_in=True)
    tgt_loss, _, _, tgt_latent = model(tgt_batch['spectrum'], 
                                       mask_ratio=mask_ratio, norm_in=True)

    # Compute total loss
    loss = src_loss + target_loss_weight*tgt_loss
    
    if mode=='train':

        # Backpropagate and update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    
        # Adjust learning rate
        lr_scheduler.step()

        # Save loss and metrics
        losses_cp['train_loss'].append(float(loss))
        losses_cp['train_src_loss'].append(float(src_loss))
        losses_cp['train_tgt_loss'].append(float(tgt_loss))
    else:
        val_feats_dist = latent_distance(src_latent, tgt_latent)
        #val_feats_dist = latent_distance(latent[:src_batch['spectrum'].size()[0]], 
        #                                latent[src_batch['spectrum'].size()[0]:])
        
        losses_cp['val_loss'].append(float(loss))
        losses_cp['val_src_loss'].append(float(src_loss))
        losses_cp['val_tgt_loss'].append(float(tgt_loss))
        losses_cp['val_feats'].append(float(val_feats_dist))
    
    return model, optimizer, lr_scheduler, losses_cp

def linear_probe_iter(model, optimizer, lr_scheduler, src_batch,
                      losses_cp, cur_iter):

    model.train(True)

    # Zero the gradients
    optimizer.zero_grad()
    
    # Compute predictions
    model_outputs = model(src_batch['spectrum'], 
                          norm_in=True, denorm_out=False, 
                          return_feats=False)

    
    # Evaluate loss on predictions vs normalized target labels
    total_loss = torch.nn.MSELoss()(model_outputs['predicted labels'], 
                                    model.normalize_labels(src_batch['stellar labels']))

    # Backpropagate and update weights
    total_loss.backward()
    optimizer.step()
    
    # Adjust learning rate
    lr_scheduler.step()

    # Save loss and metrics
    losses_cp['lp_train_loss'].append(float(total_loss))
    
    return model, optimizer, lr_scheduler, losses_cp

def linear_probe_val_iter(model, src_batch, tgt_batch, losses_cp):
        
    model.eval()
        
    # Compute prediction on source batch
    model_outputs_src = model(src_batch['spectrum'], 
                              norm_in=True, denorm_out=True, 
                              return_feats=True)
    
    # Compute prediction on target batch
    model_outputs_tgt = model(tgt_batch['spectrum'], 
                              norm_in=True, denorm_out=True, 
                              return_feats=True)
        
    
    # Compute mean absolute error on label predictions
    src_losses = []
    tgt_losses = []
    for i in range(model.num_labels):
        src_losses.append(torch.nn.L1Loss()(model_outputs_src['predicted labels'][:,i], 
                                               src_batch['stellar labels'][:,i]))
        tgt_losses.append(torch.nn.L1Loss()(model_outputs_tgt['predicted labels'][:,i], 
                                               tgt_batch['stellar labels'][:,i]))
        
    # Save losses
    for src_val, tgt_val, label_key in zip(src_losses, tgt_losses, model.label_keys):
        losses_cp['lp_val_src_'+label_key].append(float(src_val))
        losses_cp['lp_val_tgt_'+label_key].append(float(tgt_val))
        
    feat_score = float(torch.nn.L1Loss()(model_outputs_src['feature map'], 
                                   model_outputs_tgt['feature map']))
    losses_cp['lp_feat_score'].append(float(feat_score))
                        
    return losses_cp