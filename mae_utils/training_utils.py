import torch
from torch import nn
import torch.nn.functional as F
from torch._six import inf

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

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, 
                 parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

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
               losses_cp, cur_iter, total_batch_iters, mode, device):

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
    #mask_ratio = mask_ratio.item()
    
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
                      losses_cp, cur_iter, device):

    model.train_head_mode()

    # Zero the gradients
    optimizer.zero_grad()
    
    total_loss = 0.

    # Compute predictions
    #with torch.cuda.amp.autocast():
    model_outputs = model.forward_labels(src_batch['spectrum'], 
                                               norm_in=True, denorm_out=False, 
                                                 return_feats=False)

    # Compute the average loss on stellar class labels
    src_mm_loss_tot = 0.    
    if model.num_mm_labels>0:
        
        # Convert target label values to classes
        src_classes = model.multimodal_to_class(src_batch['multimodal labels'])
        for i in range(model.num_mm_labels):
            # Evaluate loss on predicted labels
            src_mm_loss = torch.nn.NLLLoss()(model_outputs['multimodal labels'][i], 
                                             src_classes[i])
            
            src_mm_loss_tot += 1/model.num_mm_labels * src_mm_loss
    # Add to total loss
    total_loss += src_mm_loss_tot

    src_um_loss_tot = 0.
    if model.num_um_labels>0:
        # Normalize target labels
        src_um_labels = model.normalize_unimodal(src_batch['unimodal labels'])
        for i in range(model.num_um_labels):
            # Evaluate loss on predictions from the entire spectra
            src_um_loss = torch.nn.MSELoss()(model_outputs['unimodal labels'][:,i], 
                                             src_um_labels[:,i])
            # Add to total loss
            src_um_loss_tot += 1/model.num_um_labels * src_um_loss
    # Add to total loss
    total_loss += src_um_loss_tot

    # Backpropagate and update weights
    total_loss.backward()
    
    optimizer.step()
    
    # Adjust learning rate
    lr_scheduler.step()

    # Save loss and metrics
    losses_cp['lp_train_loss'].append(float(total_loss))
    losses_cp['lp_train_mm_loss'].append(float(src_mm_loss_tot))
    losses_cp['lp_train_um_loss'].append(float(src_um_loss_tot))
    
    return model, optimizer, lr_scheduler, losses_cp

def linear_probe_val_iter(model, src_batch, tgt_batch, losses_cp):
        
    model.eval_mode()
        
    # Compute prediction on source batch
    model_outputs_src = model.forward_labels(src_batch['spectrum'], 
                                                   norm_in=True, denorm_out=True, 
                                                   return_feats=True)
    
    # Compute prediction on target batch
    model_outputs_tgt = model.forward_labels(tgt_batch['spectrum'], 
                                                   norm_in=True, denorm_out=True, 
                                                   return_feats=True)
        
    # Compute Mean Abs Error on multimodal label predictions
    src_mm_losses = []
    tgt_mm_losses = []
    for i in range(model.num_mm_labels):
        src_mm_losses.append(torch.nn.L1Loss()(model_outputs_src['multimodal labels'][:,i], 
                                               src_batch['multimodal labels'][:,i]))
        tgt_mm_losses.append(torch.nn.L1Loss()(model_outputs_tgt['multimodal labels'][:,i], 
                                               tgt_batch['multimodal labels'][:,i]))
    
    # Compute mean absolute error on unimodal label predictions
    src_um_losses = []
    tgt_um_losses = []
    for i in range(model.num_um_labels):
        src_um_losses.append(torch.nn.L1Loss()(model_outputs_src['unimodal labels'][:,i], 
                                               src_batch['unimodal labels'][:,i]))
        tgt_um_losses.append(torch.nn.L1Loss()(model_outputs_tgt['unimodal labels'][:,i], 
                                               tgt_batch['unimodal labels'][:,i]))
        
    # Save losses    
    for src_val, tgt_val, label_key in zip(src_mm_losses, tgt_mm_losses, model.multimodal_keys):
        losses_cp['lp_val_src_'+label_key].append(float(src_val))
        losses_cp['lp_val_tgt_'+label_key].append(float(tgt_val))
    for src_val, tgt_val, label_key in zip(src_um_losses, tgt_um_losses, model.unimodal_keys):
        losses_cp['lp_val_src_'+label_key].append(float(src_val))
        losses_cp['lp_val_tgt_'+label_key].append(float(tgt_val))
        
    feat_score = float(torch.nn.L1Loss()(model_outputs_src['feature map'], 
                                   model_outputs_tgt['feature map']))
    losses_cp['lp_feat_score'].append(float(feat_score))
                        
    return losses_cp