import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth

import os
from training_utils import str2bool
from itertools import chain
from collections import defaultdict
import math

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from pos_embed import get_2d_sincos_pos_embed


class StarNet_head_old(torch.nn.Module):
    '''
    Create a Linear output layer.
    '''
    def __init__(self, in_features, out_features=6, 
                 softplus=False, logsoftmax=False):
        super().__init__()
        
        # Fully connected layer
        layers = list()
        layers.append(torch.nn.LayerNorm(in_features))
        #layers.append(torch.nn.BatchNorm1d(in_features))
        layers.append(torch.nn.Linear(in_features, out_features))
        
        if softplus:
            layers.append(torch.nn.Softplus())
        if logsoftmax:
            layers.append(torch.nn.LogSoftmax(dim=1))
        
        self.fc_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_model(x)
    
class StarNet_head(torch.nn.Module):
    '''
    Create a Linear output layer.
    '''
    def __init__(self, in_features, out_features=6, dropout=0.0,
                 softplus=False, logsoftmax=False):
        super().__init__()
        
        # Fully connected layer
        layers = list()
        #layers.append(torch.nn.BatchNorm1d(in_features))
        #layers.append(torch.nn.LayerNorm(in_features))
        # Define proportion or neurons to dropout
        if dropout>0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(in_features, out_features))
        
        if softplus:
            layers.append(torch.nn.Softplus())
        if logsoftmax:
            layers.append(torch.nn.LogSoftmax(dim=1))
        
        self.fc_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_model(x)

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, img_dim=2, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 input_mean=None, input_std=None, device='cpu',
                multimodal_keys=None, unimodal_keys=None, mutlimodal_vals=None, 
                 unimodal_means=None, unimodal_stds=None, head_dropout=0.0, lp_enc_layers=0):
        
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.img_dim = img_dim
        self.input_mean = input_mean
        self.input_std = input_std
        self.lp_enc_layers = lp_enc_layers
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if hasattr(patch_size, '__len__'):
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[-1] * in_chans, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        
        # Network heads for linear probing
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.mutlimodal_vals = mutlimodal_vals
        self.unimodal_means = torch.tensor(unimodal_means).to(device)
        self.unimodal_stds = torch.tensor(unimodal_stds).to(device)
        self.num_mm_labels = len(self.multimodal_keys)
        self.num_um_labels = len(self.unimodal_keys)
        
        in_features = (num_patches+1)*embed_dim
        # Network head that predicts labels as linear output
        self.label_classifiers = []
        if self.num_mm_labels>0:
            for vals in mutlimodal_vals:
                # Fully connected classifier
                self.label_classifiers.append(StarNet_head(in_features=in_features, 
                                                          out_features=len(vals), 
                                                           dropout=head_dropout,
                                                          logsoftmax=False).to(device))
                '''
                self.label_classifiers.append(StarNet_head_old(in_features=in_features, 
                                                          out_features=len(vals), 
                                                           #dropout=head_dropout,
                                                          logsoftmax=True).to(device))
                '''
        
        # Network head that predicts unimodal labels as linear output
        if self.num_um_labels>0:
            self.unimodal_predictor = StarNet_head(in_features=in_features, 
                                                   out_features=self.num_um_labels,
                                                   dropout=head_dropout).to(device)
            
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        
        print(self.pos_embed.shape, pos_embed.shape, self.patch_embed.num_patches**.5)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
        if self.num_mm_labels>0:
            for net in self.label_classifiers:
                net.apply(self._init_weights)
                
        if self.num_um_labels>0:
            self.unimodal_predictor.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def multimodal_to_class(self, labels):
        '''Convert labels into classes based on the multimodal values.'''
        classes = []
        for i, vals in enumerate(self.mutlimodal_vals):
            classes.append(torch.cat([torch.where(vals==labels[j,i])[0] for j in range(len(labels))]))
        return classes
    
    def class_to_label(self, classes, take_mode=False):
        '''Convert probabilities into labels using a weighted average and the multimodal values.'''
        labels = []
        for i, (cla, c_vals) in enumerate(zip(classes, self.mutlimodal_vals)):
            
            # Turn predictions in "probabilities"
            #prob = torch.exp(cla)
            prob = torch.nn.Softmax(dim=1)(cla)
            
            if take_mode:
                # Take the class with the highest probability
                class_indices = torch.argmax(prob, dim=(1), keepdim=True)
                labels.append(torch.cat([c_vals[i] for i in class_indices]))
            else:
                # Take weighted average using class values and probabilities
                labels.append(torch.sum(prob*c_vals, axis=1))

        return torch.stack(labels).T
    
    def normalize_unimodal(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.unimodal_means) / self.unimodal_stds
    
    def denormalize_unimodal(self, labels):
        '''Rescale the labels back to their original units.'''
        return labels * self.unimodal_stds + self.unimodal_means

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def patchify1D(self, imgs):
        """
        imgs: (N, C, 1, W)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size[1]
        assert imgs.shape[2] == 1 and imgs.shape[3] % p == 0

        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, w, 1, 1, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], w, p * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def unpatchify1D(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, C, 1, W)
        """
        p = self.patch_embed.patch_size[1]
        h = 1
        w = x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], 1, w, p, 1, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, 1, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def normalize_inputs(self, x):
        '''Normalize data to have zero-mean and unit-variance.'''
        return (x - self.input_mean) / self.input_std
    
    def denormalize_inputs(self, x):
        '''Undo the normalization to put spectra back in the original scale.'''
        return x * self.input_std + self.input_mean
    
    def train_head_mode(self):
        '''Set each head to train mode.'''

        self.eval()

        # Unfreeze last couple layers of encoder for fine-tuning
        if self.lp_enc_layers>0:
            self.blocks[-1].mlp.fc2.train()
        if self.lp_enc_layers>1:
            self.blocks[-1].mlp.fc1.train()
        
        if self.num_mm_labels>0:
            for classifier in self.label_classifiers:
                classifier.train()
                
        if self.num_um_labels>0:
            self.unimodal_predictor.train()
            
    def eval_mode(self):
        '''Set each submodel to eval mode.'''
            
        self.eval()

        if self.num_mm_labels>0:
            for classifier in self.label_classifiers:
                classifier.eval()
                
        if self.num_um_labels>0:
            self.unimodal_predictor.eval()
            
    def freeze_mae(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

        # Unfreeze last couple layers of encoder for fine-tuning
        for name, param in self.blocks[-1].named_parameters():
            if ('mlp.fc2' in name) and (self.lp_enc_layers>0):
                param.requires_grad = True
            if ('mlp.fc1' in name) and (self.lp_enc_layers>1):
                param.requires_grad = True
                
        for name, param in self.unimodal_predictor.named_parameters():
            param.requires_grad = True
            
    def head_parameters(self):
        '''Create an iterable list of all network head parameters.'''
        parameters = []        

        # Include last couple layers of encoder for fine-tuning
        if self.lp_enc_layers>0:
            parameters.append(self.blocks[-1].mlp.fc2.parameters())
        if self.lp_enc_layers>1:
            parameters.append(self.blocks[-1].mlp.fc1.parameters())
        
        if self.num_mm_labels>0:
            for net in self.label_classifiers:
                parameters.append(net.parameters())
        
        if self.num_um_labels>0:
            parameters.append(self.unimodal_predictor.parameters())            
        
        return chain(*parameters)

    def forward_encoder(self, x, mask_ratio, norm_in=False):
        
        if norm_in:
            # Normalize input data
            x = self.normalize_inputs(x)
            
        if self.img_dim==1:
            x = x.reshape(shape=(x.shape[0], self.in_chans, 1, self.img_size[-1]))
        
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.img_dim==1:
            imgs = imgs.reshape(shape=(imgs.shape[0], self.in_chans, 1, self.img_size[-1]))
            target = self.patchify1D(imgs)
        else:
            target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        #loss = torch.mean((pred - target) ** 2)
        return loss

    def forward(self, imgs, mask_ratio=0.75, norm_in=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, norm_in=norm_in)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        if norm_in:
            # Normalize target data
            imgs = self.normalize_inputs(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, latent
    
    def forward_labels(self, imgs, norm_in=False, denorm_out=False, return_feats=False,
                      take_mode=False):
        
        # Encode without masking
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=0., norm_in=norm_in)
        
        return_dict = {}
            
        if return_feats:
            return_dict['feature map'] = latent
        
        latent = torch.flatten(latent, start_dim=1)
        if self.num_mm_labels>0:
            # Predict labels from features
            mm_labels = [classifier(latent) for classifier in self.label_classifiers]
            if denorm_out:
                # Denormalize labels
                mm_labels = self.class_to_label(mm_labels, take_mode=take_mode)
            return_dict['multimodal labels'] = mm_labels
                
        if self.num_um_labels>0:
            # Predict labels from features
            um_labels = self.unimodal_predictor(latent)
                
            if denorm_out:
                # Denormalize labels
                um_labels = self.denormalize_unimodal(um_labels)
            return_dict['unimodal labels'] = um_labels
        
        return return_dict

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_1D_simple(**kwargs):
    model = MaskedAutoencoderViT(img_size=(1,800), img_dim=1, patch_size=(1,50), in_chans=1,
                 embed_dim=52, depth=6, num_heads=4,
                 decoder_embed_dim=12, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def build_mae(config, device, model_name, mutlimodal_vals):
    
    # Display model configuration
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))

    # Construct Network
    print('\nBuilding networks...')
    model = MaskedAutoencoderViT(img_size=(1, int(config['MAE ARCHITECTURE']['spectrum_size'])), 
                                 img_dim=1, 
                                 patch_size=(1, int(config['MAE ARCHITECTURE']['patch_size'])), 
                                 in_chans=1,
                                 embed_dim=int(config['MAE ARCHITECTURE']['encoder_embed_dim']), 
                                 depth=int(config['MAE ARCHITECTURE']['encoder_depth']), 
                                 num_heads=int(config['MAE ARCHITECTURE']['encoder_num_heads']),
                                 decoder_embed_dim=int(config['MAE ARCHITECTURE']['decoder_embed_dim']), 
                                 decoder_depth=int(config['MAE ARCHITECTURE']['decoder_depth']), 
                                 decoder_num_heads=int(config['MAE ARCHITECTURE']['decoder_num_heads']),
                                 mlp_ratio=float(config['MAE ARCHITECTURE']['mlp_ratio']), 
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 norm_pix_loss=False,
                                 input_mean=float(config['DATA']['spectra_mean']),
                                 input_std=float(config['DATA']['spectra_std']),
                                 device=device,
                                 multimodal_keys=eval(config['DATA']['multimodal_keys']), 
                                 unimodal_keys=eval(config['DATA']['unimodal_keys']),
                                 mutlimodal_vals=mutlimodal_vals, 
                                 unimodal_means=eval(config['DATA']['unimodal_means']), 
                                 unimodal_stds=eval(config['DATA']['unimodal_stds']),
                                head_dropout=float(config['LINEAR PROBE TRAINING']['dropout']),
                                lp_enc_layers=int(config['LINEAR PROBE TRAINING']['num_enc_layers']))
    
    model.to(device)

    # Display model architectures
    print('MAE Architecture:')
    print(model)
    
    if model.num_mm_labels>0:
        print('Classifier1 Architecture:')
        print(model.label_classifiers[0])
    if model.num_um_labels>0:
        print('Linear Head Architecture:')
        print(model.unimodal_predictor)
        
    return model

#'''
def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None,
                     lp_optimizer=None, lp_lr_scheduler=None,
                     loss_scaler=None):
    
    # Check for pre-trained weights
    if os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model weights...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)
        losses = defaultdict(list, dict(checkpoint['losses']))
        cur_iter = checkpoint['batch_iters']+1
        if 'lp_batch_iters' in checkpoint.keys():
            cur_lp_iter = checkpoint['lp_batch_iters']+1
        else:
            cur_lp_iter = 1

        # Load optimizer states
        try:
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            if (lp_optimizer is not None) & (cur_lp_iter>1):
                lp_optimizer.load_state_dict(checkpoint['optimizer'])
            if (lp_lr_scheduler is not None) & (cur_lp_iter>1):
                lp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except ValueError:
            pass
        
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        try:
            if model.num_mm_labels>0:
                for net, state in zip(model.label_classifiers, checkpoint['classifier models']):
                    net.load_state_dict(state)
        except KeyError:
            pass
        
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        cur_lp_iter = 1
        
    return model, losses, cur_iter, cur_lp_iter
'''
def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None,
                     loss_scaler=None):
    
    # Check for pre-trained weights
    if os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model weights...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)
        losses = defaultdict(list, dict(checkpoint['losses']))
        cur_iter = checkpoint['batch_iters']+1
        #if 'lp_batch_iters' in checkpoint.keys():
        #    cur_lp_iter = checkpoint['lp_batch_iters']+1
        #else:
        cur_lp_iter = 1

        # Load optimizer states
        # Load model weights
        model.load_state_dict(checkpoint['model'])

    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        cur_lp_iter = 1
        
    return model, losses, cur_iter, cur_lp_iter


'''