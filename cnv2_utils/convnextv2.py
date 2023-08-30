import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy.random as random

from timm.models.layers import trunc_normal_

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    SparseTensor
)
from MinkowskiOps import (
    to_sparse,
)

from collections import OrderedDict
import math

class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, W) -> (N, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, W, C) -> (N, C, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 stem_width=4, input_mean=1, input_std=0.2, 
                 label_keys=[], label_means=[], label_stds=[], 
                 dropout=0.0, lp_enc_layers=0, device='cpu'):
        super().__init__()      
        
        self.input_mean = input_mean
        self.input_std = input_std
        self.lp_enc_layers = lp_enc_layers
        self.label_keys = label_keys
        self.label_means = torch.tensor(label_means).to(device)
        self.label_stds = torch.tensor(label_stds).to(device)
        self.num_labels = len(self.label_keys)
        self.dropout = dropout
        
        self.depths = depths
        self.stem_width = stem_width
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=stem_width, stride=stem_width),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)    
            
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head_dropout = torch.nn.Dropout(dropout)
        self.head = nn.Linear(dims[-1], self.num_labels)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def normalize_labels(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.label_means) / self.label_stds
    
    def denormalize_labels(self, labels):
        '''Rescale the labels back to their original units.'''
        return labels * self.label_stds + self.label_means
    
    def normalize_inputs(self, x):
        '''Normalize data to have zero-mean and unit-variance.'''
        return (x - self.input_mean) / self.input_std
    
    def denormalize_inputs(self, x):
        '''Undo the normalization to put spectra back in the original scale.'''
        return x * self.input_std + self.input_mean
            
    def forward_features(self, x, norm_in=True):
                
        if norm_in:
            # Normalize input data
            x = self.normalize_inputs(x)
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-1])) # global average pooling, (N, C, W) -> (N, C)

    def forward(self, x, norm_in=True, denorm_out=False, return_feats=False):
        return_dict = {}
        
        x = x.unsqueeze(1)
        x = self.forward_features(x, norm_in=norm_in)
        
        if return_feats:
            return_dict['feature map'] = latent
            
        if self.dropout>0:
            x = self.head_dropout(x)
        x = self.head(x)
        
        if denorm_out:
            # Denormalize labels
            x = self.denormalize_labels(x)
        return_dict['predicted labels'] = x
        
        return return_dict

class SparseBlock(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., D=1):
        super().__init__()
        self.dwconv = MinkowskiChannelwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)   
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4  * dim)
        self.drop_path = MinkowskiDropPath(drop_path)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 D=1,
                 stem_width=4):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.stem_width = stem_width
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=stem_width, stride=stem_width),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D)
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SparseBlock(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiChannelwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        return mask.repeat_interleave(scale, axis=1)

    def forward(self, x, mask):
        num_stages = len(self.stages)
        num_patches = mask.size()[1]
        mask = self.upsample_mask(mask, 2**(num_stages-1)) 
        mask = mask.unsqueeze(1).type_as(x)
        
        # patch embedding
        x = self.downsample_layers[0](x)
        
        x *= (1.-mask)
        
        # sparse encoding
        x = to_sparse(x)
        
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
            
        # densify
        x = x.dense()[0]
        return x

class FCMAE1D(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=224,
                in_chans=3,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                norm_pix_loss=False,
                input_mean=1.,
                input_std=0.2):
        super().__init__()

        # configs
        self.img_size = img_size
        self.in_chans = in_chans
        self.depths = depths
        self.patch_size = patch_size
        self.num_patches = (img_size / patch_size)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        
        self.input_mean = input_mean
        self.input_std = input_std
        
        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims, D=1)
        # decoder
        self.proj = nn.Conv1d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv1d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size * in_chans,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiChannelwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv1d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs):
        """
        imgs: (N, C, W)
        x: (N, L, patch_size)
        """
        p = self.patch_size
        imgs = imgs.unsqueeze(2)
        assert imgs.shape[2] == 1 and imgs.shape[3] % p == 0

        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, w, 1, 1, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], w, p * self.in_chans))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, C, 1, W)
        """
        p = self.patch_size
        h = 1
        w = x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], 1, w, p, 1, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, 1, w * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = int(x.shape[2] / self.patch_size)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1])
        return mask.reshape(-1, 1, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def normalize_inputs(self, x):
        '''Normalize data to have zero-mean and unit-variance.'''
        return (x - self.input_mean) / self.input_std
    
    def denormalize_inputs(self, x):
        '''Undo the normalization to put spectra back in the original scale.'''
        return x * self.input_std + self.input_mean
    
    def forward_encoder(self, imgs, mask_ratio, norm_in=True):
                
        if norm_in:
            imgs = self.normalize_inputs(imgs)
            
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):

        x = self.proj(x)
        # append mask token
        n, c, w = x.shape
        mask = mask.reshape(-1, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, W]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove
        """
            
        target = self.patchify(imgs)
        pred = torch.einsum('ncl->nlc', pred)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        #loss = torch.mean((pred - target) ** 2)
        return loss

    def forward(self, imgs, mask_ratio=0.6, norm_in=True):
        imgs = imgs.unsqueeze(1)
        latent, mask = self.forward_encoder(imgs, mask_ratio, norm_in=norm_in)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, latent
    
def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # reshape standard convolution
                kv, in_dim, out_dim = v.shape
                new_ckpt[new_k] = v.permute(2, 1, 0)#.\
                    #reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, kv)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
            
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        #elif 'grn' in k:
        #    new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
        
def load_sparse_to_nonsparse(model, sparse_fn):
    checkpoint = torch.load(sparse_fn, 
                            map_location=lambda storage, 
                            loc: storage)


    print("Load pre-trained checkpoint")
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token'in k or \
           'proj' in k or 'pred' in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        
    # Fix keys and parameter shapes
    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    
    # Now we can nicely load the model state dict
    load_state_dict(model, checkpoint_model)

    # Manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.constant_(model.head.bias, 0.)
    
    return model