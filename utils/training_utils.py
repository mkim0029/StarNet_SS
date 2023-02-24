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

def loss_fn(y_true, y_pred, y_sigma):
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
             source_mm_weights, source_um_weights, source_feature_weight, target_feature_weight,
             source_task_weights, target_task_weights, feat_loss_fn, losses_cp, mode='train'):
        
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
        src_mm_loss_tot = 0.  
        src_classes = model.module.multimodal_to_class(src_batch['multimodal labels'])
        for i in range(model.module.num_mm_labels):
            src_mm_loss = torch.nn.NLLLoss()(model_outputs_src['multimodal labels'][i], 
                                             src_classes[i])
            src_mm_loss_tot += 1/model.module.num_mm_labels * src_mm_loss
            # Add to total loss
            if source_mm_weights[i]>0:
                total_loss = total_loss + source_mm_weights[i]/model.module.num_mm_labels * src_mm_loss
    
    if model.module.num_um_labels>0:
        src_um_loss_tot = 0.
        src_um_labels = model.module.normalize_unimodal(src_batch['unimodal labels'])
        for i in range(model.module.num_um_labels):
            src_um_loss = torch.nn.MSELoss()(model_outputs_src['unimodal labels'][:,i], 
                                             src_um_labels[:,i])
            src_um_loss_tot += 1/model.module.num_um_labels * src_um_loss
            
            # Add to total loss
            if source_um_weights[i]>0:
                total_loss = total_loss + source_um_weights[i]/model.module.num_um_labels * src_um_loss
        
    else:
        src_um_loss_tot = 0.
        
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
        src_feature_loss = feat_loss_fn(model_outputs_src['feature map'], 
                                              model_outputs_src2['feature map'])
        
        tgt_feature_loss = feat_loss_fn(model_outputs_tgt['feature map'], 
                                              model_outputs_tgt2['feature map'])
        
        if source_feature_weight>0:
            total_loss = total_loss + source_feature_weight*src_feature_loss
        if target_feature_weight>0:
            total_loss = total_loss + target_feature_weight*tgt_feature_loss
        
    if len(model.module.tasks)>0:
        # Compute loss on task labels
        src_task_losses = task_loss_fn(model.module.normalize_tasks(src_batch['task labels']), 
                                       model_outputs_src['task labels'])
        tgt_task_losses = task_loss_fn(model.module.normalize_tasks(tgt_batch['task labels']), 
                                       model_outputs_tgt['task labels'])
        # Add to total loss
        for i in range(len(src_task_losses)):
            if source_task_weights[i]>0:
                total_loss = total_loss + 1/len(src_task_losses)*src_task_losses[i]*source_task_weights[i]
            if target_task_weights[i]>0:
                total_loss = total_loss + 1/len(tgt_task_losses)*tgt_task_losses[i]*target_task_weights[i]
        
    if mode=='train':        
        # Update the gradients
        total_loss.backward()

        # Save loss and metrics
        losses_cp['train_loss'].append(float(total_loss))
        if model.module.use_split_convs:
            losses_cp['train_src_feats'].append(float(src_feature_loss))
            losses_cp['train_tgt_feats'].append(float(tgt_feature_loss))
        if model.module.num_mm_labels>0:
            losses_cp['train_src_mm_labels'].append(float(src_mm_loss_tot))
        if model.module.num_um_labels>0:
            losses_cp['train_src_um_labels'].append(float(src_um_loss_tot))
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
    
    # Produce feature map and label predictions from source batch
    model_feats_src = []
    mm_label_preds_src = []
    um_label_preds_src = []
    for i in range(0, src_batch['spectrum chunks'].size(1), batch_size):
        model_outputs = model(src_batch['spectrum chunks'][:,i:i+batch_size].squeeze(0),
                              src_batch['pixel_indx'][:,i:i+batch_size].squeeze(0),
                              norm_in=True, denorm_out=True, return_feats=True, 
                              take_mode=True, combine_batch_probs=True)
        
        model_feats_src.append(model_outputs['feature map'])
        mm_label_preds_src.append(model_outputs['multimodal labels'])
        if model.module.num_um_labels>0:
            um_label_preds_src.append(model_outputs['unimodal labels'])
    
    # Produce feature map and label predictions from target batch
    model_feats_tgt = []
    mm_label_preds_tgt = []
    um_label_preds_tgt = []
    for i in range(0, tgt_batch['spectrum chunks'].size(1), batch_size):
        model_outputs = model(tgt_batch['spectrum chunks'][:,i:i+batch_size].squeeze(0),
                              tgt_batch['pixel_indx'][:,i:i+batch_size].squeeze(0),
                              norm_in=True, denorm_out=True, return_feats=True, 
                              take_mode=True, combine_batch_probs=True)
        
        model_feats_tgt.append(model_outputs['feature map'])
        mm_label_preds_tgt.append(model_outputs['multimodal labels'])
        if model.module.num_um_labels>0:
            um_label_preds_tgt.append(model_outputs['unimodal labels'])
        
    model_feats_src = torch.cat(model_feats_src)
    mm_label_preds_src = torch.cat(mm_label_preds_src)
    model_feats_tgt = torch.cat(model_feats_tgt)
    mm_label_preds_tgt = torch.cat(mm_label_preds_tgt)
    if model.module.num_um_labels>0:
        um_label_preds_src = torch.cat(um_label_preds_src)
        um_label_preds_tgt = torch.cat(um_label_preds_tgt)
        
    
    # Compute average from all chunks
    mm_label_preds_src = torch.mean(mm_label_preds_src, axis=0, keepdim=True)
    mm_label_preds_tgt = torch.mean(mm_label_preds_tgt, axis=0, keepdim=True)
    if model.module.num_um_labels>0:
        um_label_preds_src = torch.mean(um_label_preds_src, axis=0, keepdim=True)
        um_label_preds_tgt = torch.mean(um_label_preds_tgt, axis=0, keepdim=True)
    
    # Compute Mean Abs Error on multimodal label predictions
    src_mm_losses = []
    tgt_mm_losses = []
    for i in range(model.module.num_mm_labels):
        src_mm_losses.append(torch.nn.L1Loss()(mm_label_preds_src[0,i], 
                                               src_batch['multimodal labels'][0,i]))
        tgt_mm_losses.append(torch.nn.L1Loss()(mm_label_preds_tgt[0,i], 
                                               tgt_batch['multimodal labels'][0,i]))
        #src_mm_losses.append(torch.sqrt(torch.nn.MSELoss()(mm_label_preds_src[0,i], 
        #                                       src_batch['multimodal labels'][0,i])))
        #tgt_mm_losses.append(torch.sqrt(torch.nn.MSELoss()(mm_label_preds_tgt[0,i], 
        #                                       tgt_batch['multimodal labels'][0,i])))
    
    # Compute mean absolute error on unimodal label predictions
    src_um_losses = []
    tgt_um_losses = []
    for i in range(model.module.num_um_labels):
        src_um_losses.append(torch.nn.L1Loss()(um_label_preds_src[0,i], 
                                               src_batch['unimodal labels'][0,i]))
        tgt_um_losses.append(torch.nn.L1Loss()(um_label_preds_tgt[0,i], 
                                               tgt_batch['unimodal labels'][0,i]))
    
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
    
    # Store losses
    losses_cp['val_feats'].append(float(feat_loss))
    for src_val, tgt_val, label_key in zip(src_mm_losses, tgt_mm_losses, model.module.multimodal_keys):
        losses_cp['val_src_'+label_key].append(float(src_val))
        losses_cp['val_tgt_'+label_key].append(float(tgt_val))
    for src_val, tgt_val, label_key in zip(src_um_losses, tgt_um_losses, model.module.unimodal_keys):
        losses_cp['val_src_'+label_key].append(float(src_val))
        losses_cp['val_tgt_'+label_key].append(float(tgt_val))
    
    return losses_cp

def determine_chunk_weights(model, dataset, device):
    
    print('Determining weighting based on %i spectra...' % (len(dataset)))
    try:
        model.eval_mode()
        num_mm_labels = model.num_mm_labels
    except AttributeError:
        model.module.eval_mode()
        num_mm_labels = model.module.num_mm_labels
    
    NLL_losses = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for indx in range(len(dataset)):
            batch = dataset.__getitem__(indx)
            batch = batch_to_device(batch, device)
            
            # Collect target data
            try:
                tgt_classes = model.multimodal_to_class(batch['multimodal labels'].unsqueeze(0))
            except AttributeError:
                tgt_classes = model.module.multimodal_to_class(batch['multimodal labels'].unsqueeze(0))
            
            # Perform forward propagation
            try:
                model_outputs = model(batch['spectrum chunks'].squeeze(0), 
                                      batch['pixel_indx'].squeeze(0),
                                      norm_in=True, denorm_out=False)

            except AttributeError:
                model_outputs = model.module(batch['spectrum chunks'].squeeze(0), 
                                             batch['pixel_indx'].squeeze(0),
                                             norm_in=True, denorm_out=False)
            
            batch_losses = []
            for i in range(num_mm_labels):
                # Repeat target class for each chunk
                tgt_class = a=tgt_classes[i].repeat(model_outputs['multimodal labels'][i].shape[0])
                
                mm_loss = torch.nn.NLLLoss(reduction='none')(model_outputs['multimodal labels'][i], 
                                                             tgt_class)
                batch_losses.append(mm_loss.data.numpy())
            NLL_losses.append(batch_losses)
            
        # Take average across samples
        NLL_losses = np.array(NLL_losses)
        NLL_losses = np.mean(NLL_losses, axis=0)
        
        # Weights are inverse to the negative-log-likelihood
        chunk_weights = 1/NLL_losses
        chunk_weights /= np.sum(chunk_weights, axis=1, keepdims=True)
    return batch['pixel_indx'][:,0].data.numpy(), chunk_weights