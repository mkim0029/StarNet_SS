import torch
from torch import nn
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth

import os
from training_utils import str2bool
from itertools import chain
from collections import defaultdict
import math

def compute_out_size(in_size, mod, device):
    '''
    Compute output size of Module `mod` given an input with size `in_size`.
    '''
    
    f = mod.forward(torch.autograd.Variable(torch.Tensor(1, *in_size)).to(device))
    return f.size()[1:]

class PositionalEncoding(torch.nn.Module):
    """
    Adjusted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    to index into the positional encoding based on location of chunk in the 
    entire spectrum.
    """

    def __init__(self, d_model, spectrum_size=43480, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(spectrum_size, d_model)
        position = torch.arange(0, spectrum_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, start_indx):
        
        #x = x + self.pe[:, start_indx:start_indx+x.size()[1], :]
        #pe_batch = torch.concat([self.pe[:, i:i+x.size()[1], :] for i in start_indx])
        
        x = x + torch.concat([self.pe[:, i:i+x.size()[1], :] for i in start_indx])
        return self.dropout(x)

# Adapted from https://github.com/FrancescoSaverioZuppichini/ConvNext

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None] * x

class BottleNeckBlock(nn.Module):
    '''A residual BottleNeck block.'''
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv1d(in_features, in_features, kernel_size=7, 
                      padding=3, bias=False, groups=in_features),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv1d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv1d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x
    
class ConvNexStage(nn.Sequential):
    '''
    Stage: a collection of blocks. 
    Each stage usually downsamples the input by a factor of 2.
    This is done in the first block of the stage.
    '''
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv1d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )
        
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, stride: int):
        super().__init__(
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride),
            #nn.BatchNorm1d(out_features)
            nn.GroupNorm(num_groups=1, num_channels=out_features)
        )
        
class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        stem_filt_size: int,
        stem_stride: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features, stem_filt_size, stem_stride)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )
        

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
    
class StarNet_convs(torch.nn.Module):
    '''
    Create a sequential model of 1D Convolutional layers.
    '''
    def __init__(self, in_channels=1, num_filters=[4,16], strides=[1,1], 
                 filter_lengths=[8,8], pool_length=4, input_dropout=0.0):
        super().__init__()
        
        
        layers = list()

        # Use dropout as the first layer
        if input_dropout>0:
            layers.append(torch.nn.Dropout(input_dropout))
        
        # Convolutional layers
        for i in range(len(num_filters)):
            layers.append(torch.nn.Conv1d(in_channels, num_filters[i], 
                                          filter_lengths[i], strides[i]))
            layers.append(torch.nn.ReLU())
            in_channels=num_filters[i]

        # Avg pooling layer
        if pool_length>0:
            layers.append(torch.nn.AdaptiveAvgPool1d(pool_length))
            
        self.conv_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_model(x)
        
class StarNet_head(torch.nn.Module):
    '''
    Create a Linear output layer.
    '''
    def __init__(self, in_features, out_features=6, 
                 softplus=False, logsoftmax=False):
        super().__init__()
        
        # Fully connected layer
        layers = list()
        layers.append(torch.nn.LayerNorm(in_features))
        layers.append(torch.nn.Linear(in_features, out_features))
        
        if softplus:
            layers.append(torch.nn.Softplus())
        if logsoftmax:
            layers.append(torch.nn.LogSoftmax(dim=1))
        
        self.fc_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_model(x)
    
class StarNet(torch.nn.Module):
    def __init__(self, architecture_config, multimodal_keys,
                 unimodal_keys, mutlimodal_vals, device):
        super().__init__()
        
        # ARCHITECTURE PARAMETERS
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.mutlimodal_vals = mutlimodal_vals
        self.num_fluxes = int(architecture_config['num_fluxes'])
        spectrum_size = int(architecture_config['spectrum_size'])
        self.d_model = int(architecture_config['encoder_dim'])
        conv_widths_sh = eval(architecture_config['conv_widths_sh'])
        conv_depths_sh = eval(architecture_config['conv_depths_sh'])
        stem_features_sh = int(architecture_config['stem_features_sh'])
        stem_filt_size = int(architecture_config['stem_filt_size'])
        stem_stride = int(architecture_config['stem_stride'])
        num_filters_sp = eval(architecture_config['conv_filts_sp'])
        filter_lengths_sp = eval(architecture_config['filter_lengths_sp'])
        conv_strides_sp = eval(architecture_config['conv_strides_sp'])
        pool_length = int(architecture_config['pool_length'])
        self.unimodal_means = torch.tensor(eval(architecture_config['unimodal_means'])).to(device)
        self.unimodal_stds = torch.tensor(eval(architecture_config['unimodal_stds'])).to(device)
        self.spectra_mean = float(architecture_config['spectra_mean'])
        self.spectra_std = float(architecture_config['spectra_std'])
        self.num_mm_labels = len(self.multimodal_keys)
        self.num_um_labels = len(self.unimodal_keys)
        self.tasks = eval(architecture_config['tasks'])
        if (len(self.tasks)==1 and self.tasks[0]=='[]'):
            self.tasks = []
        self.task_means = torch.tensor(eval(architecture_config['task_means'])).to(device)
        self.task_stds = torch.tensor(eval(architecture_config['task_stds'])).to(device)
        self.device = device
          
        self.use_split_convs = len(num_filters_sp)>0   
        
        in_channels = 1
        if self.d_model>0:
            # Input layer that operates on each pixel individually
            self.encoder_inp_layer = torch.nn.Linear(in_features=in_channels, 
                                                     out_features=self.d_model)        

            # Positional encoding layer that encodes the position of the chunk along the spectrum
            self.pos_encoder = PositionalEncoding(
                d_model=self.d_model,
                dropout=0,
                spectrum_size=spectrum_size,
            )
            in_channels = self.d_model

        # Shared convolutional and pooling layers
        '''self.feature_encoder_sh = StarNet_convs(in_channels=in_channels,
                                             num_filters=num_filters_sh,
                                             strides=conv_strides_sh,
                                             filter_lengths=filter_lengths_sh, 
                                             pool_length=pool_length).to(device)'''
        self.feature_encoder_sh = ConvNextEncoder(in_channels=in_channels, 
                                                  stem_features=stem_features_sh, 
                                                  stem_filt_size=stem_filt_size, 
                                                  stem_stride=stem_stride, 
                                                  depths=conv_depths_sh, 
                                                  widths=conv_widths_sh).to(device)
        
        # Split convolutional layers
        if self.use_split_convs:
            self.feature_encoder_labels = StarNet_convs(in_channels=conv_widths_sh[-1],
                                                 num_filters=num_filters_sp,
                                                 strides=conv_strides_sp,
                                                 filter_lengths=filter_lengths_sp, 
                                                 pool_length=pool_length).to(device)
            if len(self.tasks)>0:
                self.feature_encoder_tasks = StarNet_convs(in_channels=conv_widths_sh[-1],
                                                 num_filters=num_filters_sp,
                                                 strides=conv_strides_sp,
                                                 filter_lengths=filter_lengths_sp, 
                                                 pool_length=pool_length).to(device)

        # Determine shape after convolutions have been applied
        feat_map_shape = compute_out_size((in_channels, self.num_fluxes), 
                                          self.feature_encoder_sh, device)
        if self.use_split_convs:
            feat_map_shape = compute_out_size((feat_map_shape[0], feat_map_shape[1]), 
                                             self.feature_encoder_labels, device)
        print('Feature map shape: ', feat_map_shape)

        in_features = feat_map_shape[0]*feat_map_shape[1]

        # Network head that predicts labels as linear output
        self.label_classifiers = []
        if self.num_mm_labels>0:
            for vals in mutlimodal_vals:
                # Fully connected classifier
                self.label_classifiers.append(StarNet_head(in_features=in_features, 
                                                          out_features=len(vals), 
                                                          logsoftmax=True).to(device))
        
        # Network head that predicts unimodal labels as linear output
        if self.num_um_labels>0:
            self.unimodal_predictor = StarNet_head(in_features=in_features, 
                                               out_features=self.num_um_labels).to(device)
        
        # Self-supervised task predictor
        if len(self.tasks)>0:
            self.task_predictor = StarNet_head(in_features=in_features, 
                                              out_features=len(self.tasks)).to(device)

    
    def set_multimodal_vals(self, multimodal_vals):
        self.multimodal_vals = [vals.to(self.device) for vals in multimodal_vals]
    
    def normalize_spectra(self, spectra):
        '''Normalize spectra to have zero-mean and unit-variance.'''
        return (spectra - self.spectra_mean) / self.spectra_std
    
    def denormalize_spectra(self, spectra):
        '''Undo the normalization to put spectra back in the original scale.'''
        return spectra * self.spectra_std + self.spectra_mean

    def multimodal_to_class(self, labels):
        '''Convert labels into classes based on the multimodal values.'''
        classes = []
        for i, vals in enumerate(self.mutlimodal_vals):
            classes.append(torch.cat([torch.where(vals==labels[j,i])[0] for j in range(len(labels))]))
        return classes
    
    def class_to_label(self, classes, take_mode=False):
        '''Convert probabilities into labels using a weighted sum and the multimodal values.'''
        labels = []
        for cla, c_vals in zip(classes,
                               self.mutlimodal_vals):

            #c_vals = torch.tensor(c_vals).to(cla.device)
            
            # Turn predictions in "probabilities"
            prob = torch.exp(cla)
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
    
    def normalize_tasks(self, task_labels):
        '''Normalize each task label to have zero-mean and unit-variance.'''
        return (task_labels - self.task_means) / self.task_stds
    
    def denormalize_tasks(self, task_labels):
        '''Rescale the task labels back to their original units.'''
        return task_labels * self.task_stds + self.task_means
        
    def train_mode(self):
        '''Set each submodel to train mode.'''
            
        if self.d_model>0:
            self.encoder_inp_layer.train()
            self.pos_encoder.train()

        self.feature_encoder_sh.train()
        
        if self.use_split_convs:
            self.feature_encoder_labels.train()
            if len(self.tasks)>0:
                self.feature_encoder_tasks.train()

        if self.num_mm_labels>0:
            for classifier in self.label_classifiers:
                classifier.train()
                
        if self.num_um_labels>0:
            self.unimodal_predictor.train()
            
        if len(self.tasks)>0:
            self.task_predictor.train()
            
    def eval_mode(self):
        '''Set each submodel to eval mode.'''
        
        if self.d_model>0:
            self.encoder_inp_layer.eval()
            self.pos_encoder.eval()
            
        self.feature_encoder_sh.eval()
        
        if self.use_split_convs:
            self.feature_encoder_labels.eval()
            if len(self.tasks)>0:
                self.feature_encoder_tasks.eval()

        if self.num_mm_labels>0:
            for classifier in self.label_classifiers:
                classifier.eval()
                
        if self.num_um_labels>0:
            self.unimodal_predictor.eval()
            
        if len(self.tasks)>0:
            self.task_predictor.eval()
            
    def all_parameters(self):
        '''Create an iterable list of all network parameters.'''
        parameters = []        
        
        if self.d_model>0:
            parameters.append(self.encoder_inp_layer.parameters())
            parameters.append(self.pos_encoder.parameters())
            
        parameters.append(self.feature_encoder_sh.parameters())

        if self.use_split_convs:
            parameters.append(self.feature_encoder_labels.parameters())
            if len(self.tasks)>0:
                parameters.append(self.feature_encoder_tasks.parameters())
        
        if self.num_mm_labels>0:
            for net in self.label_classifiers:
                parameters.append(net.parameters())
        
        if self.num_um_labels>0:
            parameters.append(self.unimodal_predictor.parameters())            

        if len(self.tasks)>0:
            parameters.append(self.task_predictor.parameters())
        
        return chain(*parameters)
        
    def forward(self, x, pixel_indx=None, norm_in=True, 
                denorm_out=False, take_mode=False, 
                return_feats=False, return_feats_only=False):
        
        if norm_in:
            # Normalize spectra
            x = self.normalize_spectra(x)
        
        # Add channel axis    
        x = x.unsqueeze(1)

        if self.d_model>0:
            # Pass throguh the input layer right before the encoder
            x = self.encoder_inp_layer(x.squeeze(1).unsqueeze(2))

            # Pass through the positional encoding layer
            x = self.pos_encoder(x, pixel_indx)
            x = torch.swapaxes(x, 1, 2)

        # Extract features
        x = self.feature_encoder_sh(x)

        if self.use_split_convs:
            if len(self.tasks)>0:
                x_task = self.feature_encoder_tasks(x)
                x_task = torch.flatten(x_task, start_dim=1)
            x = self.feature_encoder_labels(x)
        x = torch.flatten(x, start_dim=1)
        
        if return_feats_only:
            # Only return feature maps
            return x
        else:
            return_dict = {}
            
            if return_feats:
                return_dict['feature map'] = x
                
            if self.num_mm_labels>0:
                # Predict labels from features
                mm_labels = [classifier(x) for classifier in self.label_classifiers]
                if denorm_out:
                    # Denormalize labels
                    mm_labels = self.class_to_label(mm_labels, take_mode=take_mode)
                return_dict['multimodal labels'] = mm_labels
                
            if self.num_um_labels>0:
                # Predict labels from features
                um_labels = self.unimodal_predictor(x)
                if denorm_out:
                    # Denormalize labels
                    um_labels = self.denormalize_unimodal(um_labels)
                return_dict['unimodal labels'] = um_labels

            if len(self.tasks)>0:
                # Predict tasks
                task_labels = self.task_predictor(x_task)
                if denorm_out:
                    # Denormalize task labels
                    task_labels = self.denormalize_tasks(task_labels)
                return_dict['task labels'] = task_labels
                    
            return return_dict
        
def build_starnet(config, device, model_name, mutlimodal_vals):
    
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
    model = StarNet(config['ARCHITECTURE'], 
                    eval(config['DATA']['multimodal_keys']),
                    eval(config['DATA']['unimodal_keys']), 
                    mutlimodal_vals,
                    device)
    model.to(device)

    # Display model architectures
    if model.d_model>0:
        print('\n\nEncoder Input Layer:\n')
        print(model.encoder_inp_layer)
        print('\n\nPositional Encoder:\n')
        print(model.pos_encoder)
    print('\n\nFeature Extractor Architecture:\n')
    print(model.feature_encoder_sh)
    if model.use_split_convs:
        print(model.feature_encoder_labels)
        if len(model.tasks)>0:
            print(model.feature_encoder_tasks)
            
    if model.num_mm_labels>0:
        print('\n\nMultimodal Label Prediction Architecture:\n')
        for mod in model.label_classifiers:
            print(mod)

    if model.num_um_labels>0:
        print('\n\nUnimodal Label Prediction Architecture:\n')
        print(model.unimodal_predictor)
    if len(model.tasks)>0:
        print('\n\nTask Prediction Architecture:\n')
        print(model.task_predictor)
        
    return model

def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None):
    
    # Check for pre-trained weights
    if os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model to continue training...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)
        losses = dict(checkpoint['losses'])
        cur_iter = checkpoint['batch_iters']+1

        # Load optimizer states
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
        for net, state in zip(model.label_classifiers, checkpoint['classifier models']):
            net.load_state_dict(state)
        
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter