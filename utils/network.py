import os
import torch
from training_utils import str2bool
from itertools import chain
from collections import defaultdict
import math

# Positional encoding for CNN

def compute_out_size(in_size, mod, device):
    '''
    Compute output size of Module `mod` given an input with size `in_size`.
    '''
    
    f = mod.forward(torch.autograd.Variable(torch.Tensor(1, *in_size)).to(device))
    return f.size()[1:]

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

        # Max pooling layer
        if pool_length>0:
            layers.append(torch.nn.MaxPool1d(pool_length, pool_length))
            
        self.conv_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_model(x)
    
class StarNet_fcs(torch.nn.Module):
    '''
    Create a sequential model of Fully Connected layers.
    '''
    def __init__(self, in_features, num_hidden=[256,128]):
        super().__init__()
        
        # Fully connected layers
        layers = list()
        if len(num_hidden)>0:
            for i in range(len(num_hidden)):
                layers.append(torch.nn.Linear(in_features, num_hidden[i]))
                layers.append(torch.nn.ReLU())
                in_features = num_hidden[i]
        
        self.fc_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_model(x)
    
class StarNet_head(torch.nn.Module):
    '''
    Create a Linear output layer.
    '''
    def __init__(self, in_features, out_features=6, 
                 softplus=False, logsoftmax=False):
        super().__init__()
        
        # Fully connected layer
        layers = list()
        layers.append(torch.nn.Linear(in_features, out_features))
        
        if softplus:
            layers.append(torch.nn.Softplus())
        if logsoftmax:
            layers.append(torch.nn.LogSoftmax(dim=1))
        
        self.fc_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_model(x)
    
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
    
class StarNet(torch.nn.Module):
    def __init__(self, architecture_config, label_keys, device):
        super().__init__()
        
        # ARCHITECTURE PARAMETERS
        self.num_fluxes = int(architecture_config['num_fluxes'])
        spectrum_size = int(architecture_config['spectrum_size'])
        self.d_model = int(architecture_config['encoder_dim'])
        num_filters = eval(architecture_config['conv_filts'])
        filter_lengths = eval(architecture_config['filter_lengths'])
        conv_strides = eval(architecture_config['conv_strides'])
        pool_length = int(architecture_config['pool_length'])
        self.num_hidden = eval(architecture_config['num_hidden'])
        self.label_means = torch.tensor(eval(architecture_config['label_means'])).to(device)
        self.label_stds = torch.tensor(eval(architecture_config['label_stds'])).to(device)
        self.spectra_mean = float(architecture_config['spectra_mean'])
        self.spectra_std = float(architecture_config['spectra_std'])
        self.num_labels = len(label_keys)
        self.tasks = eval(architecture_config['tasks'])
        if (len(self.tasks)==1 and self.tasks[0]=='[]'):
            self.tasks = []
        self.task_means = torch.tensor(eval(architecture_config['task_means'])).to(device)
        self.task_stds = torch.tensor(eval(architecture_config['task_stds'])).to(device)
          
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

        # Convolutional and pooling layers
        self.feature_encoder = StarNet_convs(in_channels=in_channels,
                                             num_filters=num_filters,
                                             strides=conv_strides,
                                             filter_lengths=filter_lengths, 
                                             pool_length=pool_length).to(device)

        # Determine shape after convolutions have been applied
        feat_map_shape = compute_out_size((in_channels, self.num_fluxes), 
                                             self.feature_encoder, device)
        print('Feature map shape: ', feat_map_shape)

        if len(self.num_hidden)>0:
            # Fully connected layers
            self.feature_fcs = StarNet_fcs(in_features=feat_map_shape[0]*feat_map_shape[1], 
                                           num_hidden=self.num_hidden).to(device)
            in_features = self.num_hidden[-1] + feat_map_shape[0]*feat_map_shape[1]
        else:
            in_features = feat_map_shape[0]*feat_map_shape[1]

            
        # Network head that predicts labels as linear output
        if self.num_labels>0:
            self.label_predictor = StarNet_head(in_features=in_features, 
                                               out_features=self.num_labels).to(device)
        
        # Self-supervised task predictor
        if len(self.tasks)>0:
            self.task_predictor = StarNet_head(in_features=in_features, 
                                              out_features=len(self.tasks)).to(device)

    def normalize_spectra(self, spectra):
        '''Normalize spectra to have zero-mean and unit-variance.'''
        return (spectra - self.spectra_mean) / self.spectra_std
    
    def denormalize_spectra(self, spectra):
        '''Undo the normalization to put spectra back in the original scale.'''
        return spectra * self.spectra_std + self.spectra_mean

    def normalize_labels(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.label_means) / self.label_stds
    
    def denormalize_labels(self, labels, using_unc=False):
        '''Rescale the labels back to their original units.'''
        if using_unc:
            return labels * self.label_stds
        else:
            return labels * self.label_stds + self.label_means
    
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

        self.feature_encoder.train()
        if len(self.num_hidden)>0:
            self.feature_fcs.train()

        if self.num_labels>0:
            self.label_predictor.train()
        if len(self.tasks)>0:
            self.task_predictor.train()
            
    def eval_mode(self):
        '''Set each submodel to eval mode.'''
        
        if self.d_model>0:
            self.encoder_inp_layer.eval()
            self.pos_encoder.eval()
            
        self.feature_encoder.eval()
        if len(self.num_hidden)>0:
            self.feature_fcs.eval()

        if self.num_labels>0:
            self.label_predictor.eval()
        if len(self.tasks)>0:
            self.task_predictor.eval()
            
    def all_parameters(self):
        '''Create an iterable list of all network parameters.'''
        parameters = []
        
        if self.d_model>0:
            parameters.append(self.encoder_inp_layer.parameters())
            parameters.append(self.pos_encoder.parameters())
            
        parameters.append(self.feature_encoder.parameters())

        if len(self.num_hidden)>0:
            parameters.append(self.feature_fcs.parameters())
        
        if self.num_labels>0:
            parameters.append(self.label_predictor.parameters())            

        if len(self.tasks)>0:
            parameters.append(self.task_predictor.parameters())
        
        return chain(*parameters)
        
    def forward(self, x, pixel_indx=None, norm_in=True, 
                denorm_out=False, return_feats=False):
        
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
        x = self.feature_encoder(x)
        x = torch.flatten(x, start_dim=1)

        if len(self.num_hidden)>0:
            # Apply fully-connected layers and concatenate with features
            x = torch.cat((self.feature_fcs(x), x), -1)
        
        if return_feats:
            # Only return feature maps
            return x
        else:
            return_dict = {}
                
            if self.num_labels>0:
                # Predict labels from features
                labels = self.label_predictor(x)
                if denorm_out:
                    # Denormalize labels
                    labels = self.denormalize_labels(labels)
                return_dict['stellar labels'] = labels

            if len(self.tasks)>0:
                # Predict tasks
                task_labels = self.task_predictor(x)
                if denorm_out:
                    # Denormalize task labels
                    task_labels = self.denormalize_tasks(task_labels)
                return_dict['task_labels'] = task_labels
                    
            return return_dict
        
def build_starnet(config, device, model_name):
    
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
                    eval(config['DATA']['label_keys']), 
                    device)
    model.to(device)

    # Display model architectures
    if model.d_model>0:
        print('\n\nEncoder Input Layer:\n')
        print(model.encoder_inp_layer)
        print('\n\nPositional Encoder:\n')
        print(model.pos_encoder)
    print('\n\nFeature Extractor Architecture:\n')
    print(model.feature_encoder)
    if len(model.num_hidden)>0:
        print('\n\nFeature Fully-Connected Architecture:\n')
        print(model.feature_fcs)
    if model.num_labels>0:
        print('\n\nLabel Prediction Architecture:\n')
        print(model.label_predictor)
    if len(model.tasks)>0:
        print('\n\nTask Prediction Architecture:\n')
        print(model.task_predictor)
        
    return model

def load_model_state(model, model_filename, optimizer=None, lr_scheduler=None,
                     swa_model=None):
    
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
        if swa_model is not None:
            swa_model.load_state_dict(checkpoint['swa_model'], strict=False)

        # Load model weights
        model.load_state_dict(checkpoint['model'])
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    if swa_model is not None:
        return model, swa_model, losses, cur_iter
    else:
        return model, losses, cur_iter