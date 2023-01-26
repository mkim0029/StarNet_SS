import torch
import argparse
from torch.optim.lr_scheduler import LambdaLR
import math

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
                        help="Different data directory from ml/data.", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

# Learning rate scheduler
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, min_lr, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(self.min_lr, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
    
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    
def loss_fn(y_true, y_pred, y_sigma):
    return torch.mean(torch.log(y_sigma)/2+ (y_true - y_pred)**2/(2*y_sigma)) + 5

def task_loss_fn(y_true, y_pred):
    '''Take average of each task loss separately.'''
    return torch.mean((y_true - y_pred)**2, axis=0)

def run_iter(model, src_batch, tgt_batch, optimizer, lr_scheduler, 
             source_mm_weights, source_um_weights, source_feature_weight, target_feature_weight,
             source_task_weights, target_task_weights, losses_cp, mode='train'):
    
    if mode=='train':
        model.module.train_mode()
    else:
        model.module.eval_mode()
        
    total_loss = 0.
    # Compute prediction on source batch
    model_outputs_src = model(src_batch['spectrum'],
                              src_batch['pixel_indx'],
                              norm_in=True, denorm_out=False, return_feats=True)
    # Compute prediction on target batch
    model_outputs_tgt = model(tgt_batch['spectrum'],
                              tgt_batch['pixel_indx'],
                              norm_in=True, denorm_out=False, return_feats=True)
        
    if model.module.num_mm_labels>0:
        # Compute average loss on stellar class labels
        src_mm_loss = 0.  
        src_classes = model.module.multimodal_to_class(src_batch['multimodal labels'])
        for i in range(model.module.num_mm_labels):
            src_mm_loss = src_mm_loss + source_mm_weights[i] * torch.nn.NLLLoss()(model_outputs_src['multimodal labels'][i], 
                                                                                 src_classes[i])
        src_mm_loss = src_mm_loss * 1/model.module.num_mm_labels
        # Add to total loss
        total_loss = total_loss + src_mm_loss
    
    if model.module.num_um_labels>0:
        src_um_loss = torch.nn.MSELoss()(model_outputs_src['unimodal labels'], 
                                            model.module.normalize_unimodal(src_batch['unimodal labels']))
        
        # Add to total loss
        total_loss = total_loss + src_um_loss
    else:
        src_label_loss = 0.
        
    if model.module.use_split_convs:
        # Compute prediction on second source batch
        model_outputs_src2 = model(src_batch['spectrum2'],
                                  src_batch['pixel_indx2'],
                                  norm_in=True, denorm_out=False, return_feats=True)
        # Compute prediction on second target batch
        model_outputs_tgt2 = model(tgt_batch['spectrum2'],
                                  tgt_batch['pixel_indx2'],
                                  norm_in=True, denorm_out=False, return_feats=True)
        
        # Compare feature maps
        src_feature_loss = torch.nn.MSELoss()(model_outputs_src['feature map'], 
                                              model_outputs_src2['feature map'])
        
        tgt_feature_loss = torch.nn.MSELoss()(model_outputs_tgt['feature map'], 
                                              model_outputs_tgt2['feature map'])
        
        
        total_loss = total_loss + torch.mean(source_feature_weight*src_feature_loss)
        total_loss = total_loss + torch.mean(target_feature_weight*tgt_feature_loss)
        
    if len(model.module.tasks)>0:
        # Compute loss on task labels
        src_task_losses = task_loss_fn(model.module.normalize_tasks(src_batch['task labels']), 
                                       model_outputs_src['task labels'])
        tgt_task_losses = task_loss_fn(model.module.normalize_tasks(tgt_batch['task labels']), 
                                       model_outputs_tgt['task labels'])
        # Add to total loss
        total_loss = total_loss + torch.mean(src_task_losses*source_task_weights)
        total_loss = total_loss + torch.mean(tgt_task_losses*target_task_weights)
        
    if mode=='train':        
        # Update the gradients
        total_loss.backward()

        # Save loss and metrics
        losses_cp['train_loss'].append(float(total_loss))
        if model.module.use_split_convs:
            losses_cp['train_src_feats'].append(float(src_feature_loss))
            losses_cp['train_tgt_feats'].append(float(tgt_feature_loss))
        if model.module.num_mm_labels>0:
            losses_cp['train_src_mm_labels'].append(float(src_mm_loss))
        if model.module.num_um_labels>0:
            losses_cp['train_src_um_labels'].append(float(src_um_loss))
        if len(model.module.tasks)>0:
            losses_cp['train_src_tasks'].append(src_task_losses.cpu().data.numpy().tolist())
            losses_cp['train_tgt_tasks'].append(tgt_task_losses.cpu().data.numpy().tolist())

        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        # Adjust learning rate
        lr_scheduler.step()

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(total_loss))
        if model.module.num_labels>0:
            losses_cp['val_src_labels'].append(float(src_label_loss))
        if len(model.module.tasks)>0:
            losses_cp['val_src_tasks'].append(src_task_losses.cpu().data.numpy().tolist())
            losses_cp['val_tgt_tasks'].append(tgt_task_losses.cpu().data.numpy().tolist())
                
    return model, optimizer, lr_scheduler, losses_cp

def compare_val_sample(model, src_batch, tgt_batch, losses_cp, batch_size=16):
    
    model.module.eval_mode()
    
    # Produce feature map of source batch
    model_feats_src = []
    for i in range(0, src_batch['spectrum chunks'].size(1), batch_size):
        model_feats_src.append(model(src_batch['spectrum chunks'][:,i:i+batch_size].squeeze(0),
                                     src_batch['pixel_indx'][:,i:i+batch_size].squeeze(0),
                                     norm_in=True, return_feats_only=True))
    
    # Produce feature map of target batch
    model_feats_tgt = []
    for i in range(0, tgt_batch['spectrum chunks'].size(1), batch_size):
        model_feats_tgt.append(model(tgt_batch['spectrum chunks'][:,i:i+batch_size].squeeze(0),
                                     tgt_batch['pixel_indx'][:,i:i+batch_size].squeeze(0),
                                     norm_in=True, return_feats_only=True))
    model_feats_src = torch.cat(model_feats_src)
    model_feats_tgt = torch.cat(model_feats_tgt)
    
    # Predict labels
    mm_label_preds_src = [classifier(model_feats_src) for classifier in model.module.label_classifiers]
    mm_label_preds_tgt = [classifier(model_feats_tgt) for classifier in model.module.label_classifiers]
    um_label_preds_src = model.module.unimodal_predictor(model_feats_src)
    um_label_preds_tgt = model.module.unimodal_predictor(model_feats_tgt)
    
    # Compute average from all chunks
    mm_label_preds_src = [torch.mean(preds, axis=0, keepdim=True) for preds in mm_label_preds_src]
    mm_label_preds_tgt = [torch.mean(preds, axis=0, keepdim=True) for preds in mm_label_preds_tgt]
    um_label_preds_src = torch.mean(um_label_preds_src, axis=0)
    um_label_preds_tgt = torch.mean(um_label_preds_tgt, axis=0)
    
    # Compute Mean Abs Error on multimodal label predictions
    src_mm_losses = []
    tgt_mm_losses = []
    mm_label_preds_src = model.module.class_to_label(mm_label_preds_src)
    mm_label_preds_tgt = model.module.class_to_label(mm_label_preds_tgt)
    for i in range(model.module.num_mm_labels):
        src_mm_losses.append(torch.nn.L1Loss()(mm_label_preds_src[0,i], 
                                               src_batch['multimodal labels'][0,i]))
        tgt_mm_losses.append(torch.nn.L1Loss()(mm_label_preds_tgt[0,i], 
                                               tgt_batch['multimodal labels'][0,i]))
    
    # Compute mean squared error on unimodal label predictions
    src_um_loss = torch.nn.MSELoss()(um_label_preds_src, 
                                     model.module.normalize_unimodal(src_batch['unimodal labels'][0]))
    tgt_um_loss = torch.nn.MSELoss()(um_label_preds_tgt, 
                                     model.module.normalize_unimodal(tgt_batch['unimodal labels'][0]))
    
    # Compute max and min of each feature
    max_feat = torch.max(torch.cat((model_feats_src, model_feats_tgt), 0), 
                         dim=0).values
    min_feat = torch.min(torch.cat((model_feats_src, model_feats_tgt), 0), 
                         dim=0).values

    # Normalize each feature between 0 and 1 across the entire batch
    model_feats_src_norm = (model_feats_src-min_feat)/(max_feat-min_feat+1e-8)
    model_feats_tgt_norm = (model_feats_tgt-min_feat)/(max_feat-min_feat+1e-8)
    
    # Find aligned chunks between source and target
    src_indices = []
    for i in range(tgt_batch['pixel_indx'].size()[1]):
        src_indices.append(torch.where(src_batch['pixel_indx'][0,:,0]==tgt_batch['pixel_indx'][0,i,0])[0])
    src_indices = torch.cat(src_indices)
    
    # Compute mean absolute error
    feat_loss = torch.mean(torch.abs(model_feats_src_norm[src_indices]-model_feats_tgt_norm))
    
    losses_cp['val_src_um'].append(float(src_um_loss))
    losses_cp['val_tgt_um'].append(float(tgt_um_loss))
    losses_cp['val_feats'].append(float(feat_loss))
    
    for src_val, tgt_val, label_key in zip(src_mm_losses, tgt_mm_losses, model.module.multimodal_keys):
        losses_cp['val_src_'+label_key].append(float(src_val))
        losses_cp['val_tgt_'+label_key].append(float(tgt_val))
    
    return losses_cp