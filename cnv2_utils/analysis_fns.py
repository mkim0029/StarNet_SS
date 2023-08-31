import numpy as np
import torch
import h5py

import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_loader import batch_to_device

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines

import seaborn as sns

from string import ascii_lowercase

from scipy import stats
import scipy.optimize as opt

from sklearn.manifold import TSNE

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

def plot_progress(losses, y_lims=[(0,1)], x_lim=None, lp=False,
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = 1
    if ((not lp) & ('val_feats' in losses.keys())):
        num_ax += 1
    elif (lp & ('lp_feat_score' in losses.keys())):
        num_ax += 1

        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    ax1 = plt.subplot(gs[0])
    
    axs = [ax1]
    
    ax1.set_title('(a) Objective Function', fontsize=fontsize)
    if lp:
        ax1.plot(losses['lp_batch_iters'], losses['lp_train_loss'],
                 label=r'Train', c='r')
    else:
        ax1.plot(losses['batch_iters'], losses['train_loss'],
                 label=r'Train', c='r')
        ax1.plot(losses['batch_iters'], losses['val_loss'],
                 label=r'Val', c='k')
    '''ax1.plot(losses['batch_iters'], losses['train_src_loss'],
             label=r'Source', c='r')
    ax1.plot(losses['batch_iters'], losses['train_tgt_loss'],
             label=r'Target', c='g')'''
    ax1.set_ylabel('Loss',fontsize=fontsize)
    
    if num_ax>1: 
        if ((not lp) & ('val_feats' in losses.keys())):
            iters = losses['batch_iters']
            feat_score = np.array(losses['val_feats'])
        elif (lp & ('lp_feat_score' in losses.keys())):
            iters = losses['lp_batch_iters']
            feat_score = np.array(losses['lp_feat_score'])
        ax = plt.subplot(gs[-1])   
        axs.append(ax)
        ax.set_title('(b) Feature Matching Score', 
                         fontsize=fontsize)
        ax.plot(iters, feat_score,
                label=r'Validation', c='k')
        ax.set_ylabel('Distance',fontsize=fontsize)

        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)
    
    for i, ax in enumerate(axs):
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            if lp:
                ax.set_xlim(losses['lp_batch_iters'][0], losses['lp_batch_iters'][-1])
            else:
                ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        ax.set_ylim(*y_lims[i])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)
        ax.legend(fontsize=fontsize_small, ncol=1)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_5_samples(wave_grid, x_orig, x_mask, x_pred, 
                    labels=[r'$x_{orig}$', r'$x_{inp}$', r'$x_{pred}$'],
                  savename=None):
    # Calculate residulal
    x_resid = (x_orig-x_pred)
    
    # Select a random 5 samples
    test_indices = np.random.uniform(0, len(x_orig), size=5).astype(int)

    plt.close('all')
    # Plot test results
    fig, axes = plt.subplots(20,1,figsize=(18, 15), sharex=True)
    for i, indx in enumerate(test_indices):
        orig, = axes[i*4].plot(wave_grid, x_orig[indx],c='k')
        axes[i*4].set_ylim((0.5,1.2))
        msk, = axes[i*4+1].plot(wave_grid, x_mask[indx],c='r')
        axes[i*4+1].set_ylim((0.5,1.2))
        pred, = axes[4*i+2].plot(wave_grid, x_pred[indx],c='b')
        axes[4*i+2].set_ylim((0.5,1.2))
        resid, = axes[4*i+3].plot(wave_grid, x_resid[indx],c='g')
        axes[4*i+3].set_ylim((-0.15,0.15))
    plt.subplots_adjust(right=0.85)
    fig.legend([orig, msk, pred, resid],
               [labels[0],labels[1],labels[2],labels[0]+r'$ - $'+labels[2]],
              loc='center right', fontsize=20)  
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_spec_resid_density(wave_grid, x_orig, x_mask, x_pred,  
                            ylim, hist=True, kde=True,
                            dist_bins=180, hex_grid=300, bias='med', scatter='std',
                            bias_label=r'$\overline{{m}}$ \ ',
                            scatter_label=r'$s$ \ ',
                            cmap="ocean_r", savename=None):
    
    x_pred[~np.isnan(x_mask)] = np.nan
    
    # Calculate residulal
    x_resid = (x_orig-x_pred)
    
    bias = np.nanmean(x_resid)
    scatter = np.nanstd(x_resid)
   
    fig = plt.figure(figsize=(15, 5)) 
    gs = gridspec.GridSpec(1, 2,  width_ratios=[5., 1])
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1], sharey=ax0)

    a = ax0.hexbin(wave_grid, x_resid, gridsize=hex_grid, cmap=cmap,  bins='log')
    cmax = np.max(a.get_array())
    ax0.set_xlim(wave_grid[0], wave_grid[-1])
    ax0.tick_params(axis='y',
                    labelsize=25,width=1,length=10)
    ax0.tick_params(axis='x',
                    bottom=True,
                    labelbottom=True,
                    labelsize=25,width=1,length=10)
    ax0.set_xlabel(r'Wavelength (\AA)',fontsize=30)
    ax0.set_ylim(ylim)

    sns.kdeplot(y=x_resid.flatten(), 
                ax=ax1, lw=3, c=a.cmap(cmax/4.), gridsize=dist_bins)
    ax1.set_xticks([])
    ax1.tick_params(axis='x',          
                    which='both',     
                    bottom=False,      
                    top=False,         
                    labelbottom=False)   
    ax1.tick_params(axis='y',          
                    which='both',   
                    left=False,     
                    right=True,        
                    labelleft=False,
                    labelright=True,
                    labelsize=25,width=1,length=10)
    ax1.set_ylim(ylim)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    ax0.annotate(r'%s=\ %0.4f \ \ %s=\ %0.4f'%(bias_label, bias, 
                                                        scatter_label, scatter),
                 xy=(0.3, 0.87), xycoords='axes fraction', fontsize=25, bbox=bbox_props)

    cax = fig.add_axes([0.86, 0.15, .015, 0.72])
    cb = plt.colorbar(a, cax=cax)
    cb.set_label(r'Count', size=30)
    cb.ax.tick_params(labelsize=25,width=1,length=10) 
    fig.subplots_adjust(wspace=0.01, bottom=0.3, right=0.78)
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_val_MAEs(losses, multimodal_keys, y_lims=[(0,1)], x_lim=None, 
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = len(multimodal_keys)
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    
    linestyles = ['-', '--', '-.', ':']
    
    for i, key in enumerate(multimodal_keys):
        
        ax = plt.subplot(gs[i])
        
        # Make label pretty
        label_key = key
        if key=='teff':
            label_key = 'T$_{\mathrm{eff}}$ [K]'
        if key=='feh':
            label_key = '[Fe/H]'
        if key=='logg':
            label_key = '$\log{g}$'
        if key=='alpha':
            label_key = r'[$\alpha$/H]'
        if key=='vrad':
            label_key = r'$v_{\mathrm{rad}}$ [km/s]'
    
        ax.set_title('(%s) %s' % (ascii_lowercase[i], label_key), fontsize=fontsize)
        ax.plot(losses['lp_batch_iters'], losses['lp_val_src_%s' % key],
                 label=r'Source', c='k')
        ax.plot(losses['lp_batch_iters'], losses['lp_val_tgt_%s' % key],
                 label=r'Target', c='r')
        ax.set_ylabel('MAE',fontsize=fontsize)
        ax.set_ylim(*y_lims[i])
        ax.legend(fontsize=fontsize_small, ncol=2)

        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['lp_batch_iters'][0], losses['lp_batch_iters'][-1])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()

def plot_resid_violinplot(label_keys, tgt_stellar_labels, pred_stellar_labels, 
                       sigma_stellar_labels=None,
                       y_lims = [1000, 1, 1, 1, 10], savename=None):
    fig, axes = plt.subplots(len(label_keys), 1, figsize=(10, len(label_keys)*2.7))

    #if not hasattr(axes, 'len'):
    #    axes = [axes]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]
        # Make label pretty
        if label_key=='teff':
            label_key = 'T$_{\mathrm{eff}}$ [K]'
        if label_key=='feh':
            label_key = '[Fe/H]'
        if label_key=='logg':
            label_key = '$\log{g}$'
        if label_key=='alpha':
            label_key = r'[$\alpha$/H]'
        if label_key=='vrad':
            label_key = r'$v_{\mathrm{rad}}$ [km/s]'
            
        # Calculate residual
        diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
        if sigma_stellar_labels is not None:

            uncts = sigma_stellar_labels[:,i]
            
            max_uncts = 0.5*y_lims[i]
            pts = ax.scatter(tgt_stellar_labels[:,i], 
                        pred_stellar_labels[:,i] - tgt_stellar_labels[:,i], 
                        alpha=0.5, s=20, zorder=1, c=sigma_stellar_labels[:,i], 
                       vmin=0, vmax=max_uncts)
            if 'eff' in label_key:
                ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.0f' % (max_uncts),
                            (0.67,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            elif 'rad' in label_key:
                ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.1f' % (max_uncts),
                            (0.62,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            else:
                ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.2f' % (max_uncts),
                            (0.6,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
        else:
            
            box_positions = []
            box_data = []
            for tgt_val in np.unique(tgt_stellar_labels[:,i]):
                indices = np.where(tgt_stellar_labels[:,i]==tgt_val)[0]
                if len(indices)>2:
                    box_positions.append(tgt_val)
                    box_data.append(diff[indices])
            box_width = np.mean(np.diff(box_positions))/2
            
            ax.violinplot(box_data, positions=box_positions, widths=box_width,
                          showextrema=True, showmeans=False)
            if 'eff' in label_key:
                ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.median(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            elif 'rad' in label_key:
                ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.median(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            else:
                ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.median(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        ax.set_xlabel('%s' % (label_key), size=4*len(label_keys))
        ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(label_keys))
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_xlim(np.min(box_positions)-box_width*2, np.max(box_positions)+box_width*2)
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])
        
        ax.text(box_positions[0]-2*box_width, 1.11*y_lims[i], 'n = ',
               fontsize=3*len(label_keys))
        ax_t = ax.secondary_xaxis('top')
        ax_t.set_xticks(box_positions)
        ax_t.set_xticklabels([len(d) for d in box_data])
        ax_t.tick_params(axis='x', direction='in', labelsize=3*len(label_keys))
        '''
        for p, d in zip(box_positions, box_data):
            ax.text(p, 0.7*y_lims[i], len(d))
        '''
        if 'eff' in label_key:
            ax.set_xticks(box_positions)
            ax.set_xticklabels(np.array(box_positions).astype(int))

        ax.tick_params(labelsize=3*len(label_keys))
        ax.grid()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=.5)

    if sigma_stellar_labels is not None:
        plt.subplots_adjust(right=0.88)
        cax = plt.axes([0.9, 0.07, 0.04, 0.9])
        cbar = plt.colorbar(pts, cax=cax, ticks=[0, max_uncts])
        cbar.ax.set_yticklabels(['0', '$>\sigma_{max}$'], size=4*len(label_keys))
        
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()

def plot_resid(label_keys, tgt_stellar_labels, pred_stellar_labels,
               sigma_stellar_labels=None, x_label='',
              y_lims = [1000, 1, 1, 1, 10], savename=None):
    fig, axes = plt.subplots(len(label_keys), 1, figsize=(10, len(label_keys)*2.5))

    #if not hasattr(axes, 'len'):
    #    axes = [axes]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]
        # Make label pretty
        if label_key=='teff':
            label_key = 'T$_{\mathrm{eff}}$ [K]'
        if label_key=='feh':
            label_key = '[Fe/H]'
        if label_key=='logg':
            label_key = '$\log{g}$'
        if label_key=='alpha':
            label_key = r'[$\alpha$/H]'
        if label_key=='vrad':
            label_key = r'$v_{\mathrm{rad}}$ [km/s]'
            
        # Calculate residual
        diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
        if sigma_stellar_labels is not None:

            uncts = sigma_stellar_labels[:,i]
            
            max_uncts = 0.5*y_lims[i]
            pts = ax.scatter(tgt_stellar_labels[:,i], 
                        pred_stellar_labels[:,i] - tgt_stellar_labels[:,i], 
                        alpha=0.5, s=20, zorder=1, c=sigma_stellar_labels[:,i], 
                       vmin=0, vmax=max_uncts)
            if 'eff' in label_key:
                ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.0f' % (max_uncts),
                            (0.67,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            elif 'rad' in label_key:
                ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.1f' % (max_uncts),
                            (0.62,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            else:
                ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)) + 
                             ' $\sigma_{max}$=%0.2f' % (max_uncts),
                            (0.6,0.82), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
        else:
            ax.scatter(tgt_stellar_labels[:,i], 
                        pred_stellar_labels[:,i] - tgt_stellar_labels[:,i], 
                        alpha=0.5, s=5, zorder=1, c='maroon')
            if 'eff' in label_key:
                ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.mean(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            elif 'rad' in label_key:
                ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.mean(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            else:
                ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        ax.set_xlabel('%s %s' % (x_label, label_key), size=4*len(label_keys))
        ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(label_keys))
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])

        ax.tick_params(labelsize=2.8*len(label_keys))
        ax.grid()
    
    plt.tight_layout()

    if sigma_stellar_labels is not None:
        plt.subplots_adjust(right=0.88)
        cax = plt.axes([0.9, 0.07, 0.04, 0.9])
        cbar = plt.colorbar(pts, cax=cax, ticks=[0, max_uncts])
        cbar.ax.set_yticklabels(['0', '$>\sigma_{max}$'], size=4*len(label_keys))
        
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()
    
def plot_resid_hexbin(label_keys, tgt_stellar_labels, pred_stellar_labels,
                      x_label='', y_lims=[1000, 1, 1, 1, 10], 
                      gridsize=(100,50), max_counts=30, cmap='ocean_r', n_std=3,
                      savename=None):
    
    fig, axes = plt.subplots(len(label_keys), 1, 
                             figsize=(10, len(label_keys)*2.5))

    #if not hasattr(axes, 'len'):
    #    axes = [axes]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]
        # Make label pretty
        if label_key=='teff':
            label_key = 'T$_{\mathrm{eff}}$ [K]'
        if label_key=='feh':
            label_key = '[Fe/H]'
        if label_key=='logg':
            label_key = '$\log{g}$'
        if label_key=='alpha':
            label_key = r'[$\alpha$/H]'
        if label_key=='vrad':
            label_key = r'$v_{\mathrm{rad}}$ [km/s]'
            
        # Calculate residual
        diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
        
        # Plot
        tgts = tgt_stellar_labels[:,i]
        x_range = [np.max([np.min(tgts), np.median(tgts)-n_std*np.std(tgts)]),
                   np.min([np.max(tgts), np.median(tgts)+n_std*np.std(tgts)])]
        
        hex_data = ax.hexbin(tgt_stellar_labels[:,i], diff, gridsize=gridsize, cmap=cmap,
                                 extent=(x_range[0], x_range[1], -y_lims[i], y_lims[i]), 
                                 bins=None, vmax=max_counts) 
        
        # Annotate with statistics
        if 'eff' in label_key:
            ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.mean(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        elif 'rad' in label_key:
            ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.mean(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        else:
            ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)),
                    (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                    bbox=bbox_props)
            
        # Axis params
        ax.set_xlabel('%s %s' % (x_label, label_key), size=4*len(label_keys))
        ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(label_keys))
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])

        ax.tick_params(labelsize=2.8*len(label_keys))
        ax.grid()
    
    # Colorbar
    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(hex_data, cax=cbar_ax)
    cbar.set_label('Counts', size=4*len(label_keys))
    
    #plt.tight_layout()
        
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()
    
def compare_veracity(isochrone_fn, teff1, logg1, feh1, teff2, logg2, feh2,
                     label1, label2,
                     feh_min, feh_max, feh_lines=[-1., -0.5, 0.], 
                     savename=None):
    
    # Load isochrone data
    with h5py.File(isochrone_fn, 'r') as f:
        iso_teff = f['teff'][:]
        iso_logg = f['logg'][:]
        iso_feh = f['feh'][:]
    
    fig = plt.figure(figsize=(16.5,4.5))

    gs = gridspec.GridSpec(1, 25, figure=fig)

    ax1 = plt.subplot(gs[:,:11])
    ax2 = plt.subplot(gs[:,13:24])
    ax1.set_title('(a) %s' % label1, fontsize=20)
    ax2.set_title('(b) %s' % label2, fontsize=20)

    pts = ax1.scatter(teff1, logg1, c=feh1, s=1, cmap='viridis_r', vmin=feh_min, vmax=feh_max)
    ax2.scatter(teff2, logg2, c=feh2, s=1, cmap='viridis_r', vmin=feh_min, vmax=feh_max)
    
    for ax in [ax1,ax2]:
        ax.set_xlim((7000,2500))
        ax.set_ylim((5.5,-0.5))
        ax.tick_params(labelsize=14)
        ax.set_xlabel('T$_{\mathrm{eff}}$ [K]', fontsize=18)
        ax.set_ylabel('$\log{g}$', fontsize=18)
        
        # Plot isochrones
        # Plot isochrones
        for fe, ls in zip(feh_lines, ['--', '-', ':', '-.']):
            ax.plot(iso_teff[iso_feh==fe], iso_logg[iso_feh==fe], 
                    ls, c='black', lw=2, label='[Fe/H] = %s'%fe)
        
    if len(feh_lines)<4:
        ax1.legend(fontsize=14, loc=[0.05,0.7], framealpha=0.5)
    else:
        ax1.legend(fontsize=14, loc=[0.05,0.65], framealpha=0.5)
    
    # Colorbar
    cax = plt.subplot(gs[:,-1])
    c_ticks = np.linspace(feh_min, feh_max, 5)
    c_ticklabels = np.round(c_ticks,1).astype(str)
    c_ticklabels[0] = r'$<$'+c_ticklabels[0]
    c_ticklabels[-1] = r'$>$'+c_ticklabels[-1]
    cbar = plt.colorbar(pts, cax=cax)
    cax.set_yticks(c_ticks, labels=c_ticklabels)
    cax.tick_params(labelsize=14)
    cax.set_ylabel('[Fe/H]', fontsize=18)

    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()
    
def plot_one_to_one(label_keys, tgt_stellar_labels, pred_stellar_labels):

    fig, axes = plt.subplots(len(label_keys), 1, figsize=(6, len(label_keys)*6))

    for i, ax in enumerate(axes):
        if np.min(tgt_stellar_labels[:,i])<0:
            label_min = np.min(tgt_stellar_labels[:,i])*1.1
        else:
            label_min = np.min(tgt_stellar_labels[:,i])*0.9
        label_max = np.max(tgt_stellar_labels[:,i])*1.1
        ax.scatter(tgt_stellar_labels[:,i], 
                    pred_stellar_labels[:,i], 
                    alpha=0.5, s=5, zorder=1, c='maroon')
        ax.set_xlabel(r'Target %s' % label_keys[i], size=4*len(label_keys))
        ax.set_ylabel(r'Predicted %s' % label_keys[i], size=4*len(label_keys))
        ax.plot([label_min, label_max], [label_min, label_max],'k--')
        if 'vrad' in label_keys[i]:
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
        else:
            ax.set_xlim(label_min, label_max)
            ax.set_ylim(label_min, label_max)

    plt.tight_layout()
    plt.show()
    
def mae_predict(model, dataloader, device, mask_ratio=0.75):
    
    print('Predicting on %i batches...' % (len(dataloader)))
    try:
        model.eval()
    except AttributeError:
        model.module.eval()

    pred_spectra = []
    mask_spectra = []
    orig_spectra = []
    latents = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for batch in dataloader:
            
            batch = batch_to_device(batch, device)
            
            loss, spec_pred, mask, latent = model.forward(batch['spectrum'], 
                                                          mask_ratio=mask_ratio, 
                                                          norm_in=True)
            
            # Construct full-sized mask
            mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size)
            mask = mask.reshape(mask.shape[0], -1).data.cpu().numpy()    
            
            # Collect original spectrum
            orig_spec = batch['spectrum'].data.cpu().numpy()
            
            # Fill in missing prediction pixels with original flux values
            spec_pred = spec_pred.reshape(spec_pred.shape[0], -1)
            spec_pred = model.denormalize_inputs(spec_pred).data.cpu().numpy()
            spec_pred[mask==0] = orig_spec[mask==0]
            
            # Representation of masked spectra
            input_spec = np.copy(orig_spec)
            input_spec[mask==1] = np.nan
            
            # Save results
            pred_spectra.append(spec_pred)
            mask_spectra.append(input_spec)
            orig_spectra.append(orig_spec)
            latents.append(latent.data.cpu().numpy())
        pred_spectra = np.concatenate(pred_spectra)
        mask_spectra = np.concatenate(mask_spectra)
        orig_spectra = np.concatenate(orig_spectra)
        latents = np.concatenate(latents)
        
    return pred_spectra, mask_spectra, orig_spectra, latents

def encoder_predict(model, dataloader, device, mask_ratio=0.):
    
    print('Predicting on %i batches...' % (len(dataloader)))
    try:
        model.eval()
    except AttributeError:
        model.module.eval()

    latents = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for batch in dataloader:
            
            batch = batch_to_device(batch, device)
            
            latent, mask = model.forward_encoder(batch['spectrum'],  
                                                 mask_ratio=mask_ratio, 
                                                 norm_in=True)
            # Save results
            latents.append(latent.data.cpu().numpy())
        latents = np.concatenate(latents)
        
    return latents

def predict_labels(model, dataloader, device, batchsize=16):
    
    print('Predicting on %i batches...' % (len(dataloader)))
    try:
        model.eval_mode()
    except AttributeError:
        model.module.eval_mode()

    tgt_labels = []
    pred_labels = []
    feature_maps = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for batch in dataloader:
            
            batch = batch_to_device(batch, device)

            # Collect target data
            tgt_labels.append(batch['stellar labels'].data.cpu().numpy())

            # Perform forward propagation
            try:
                # Compute prediction on source batch
                model_outputs = model(batch['spectrum'], 
                                                   norm_in=True, denorm_out=True, 
                                                   return_feats=True)
            except AttributeError:
                model_outputs = model.module(batch['spectrum'], 
                                                   norm_in=True, denorm_out=True, 
                                                   return_feats=True)

                
            # Save predictions
            feature_maps.append(model_outputs['feature map'].data.cpu().numpy())
            pred_labels.append(model_outputs['predicted labels'].data.cpu().numpy())

        feature_maps = np.vstack(feature_maps)
        
        
    return model.label_keys, np.concatenate(tgt_labels), np.concatenate(pred_labels), feature_maps

def run_tsne(data_a, data_b, perplex):

    m = len(data_a)

    # Combine data into a single array
    t_data = np.row_stack((data_a,data_b))
    
    # Convert data to float64 matrix. float64 is need for bh_sne
    t_data = np.asarray(t_data).astype('float64')
    t_data = t_data.reshape((t_data.shape[0], -1))
    
    # Run t-SNE    
    vis_data = TSNE(n_components=2, 
                    perplexity=perplex).fit_transform(t_data)
    
    # Separate 2D into x and y axes information
    vis_x_a = vis_data[:m, 0]
    vis_y_a = vis_data[:m, 1]
    vis_x_b = vis_data[m:, 0]
    vis_y_b = vis_data[m:, 1]
    
    return vis_x_a, vis_y_a, vis_x_b, vis_y_b

def tsne_comparison(data1, data2, 
                    perplex=80, 
                    label1=r'$\mathbf{\mathcal{X}_{synth}}$',
                    label2=r'$\mathbf{\mathcal{X}_{obs}}$',
                    savename=None):
    # Perform t-SNE on a subsample of the data
    tx_1, ty_1, tx_2, ty_2 = run_tsne(data1, 
                                      data2, 
                                      perplex=perplex)

    # Plot them together
    plt.figure(figsize=(6,6))
    plt.scatter(tx_1, ty_1,
                label=label1,
                marker='o', c='cornflowerblue', alpha=0.2)
    plt.scatter(tx_2, ty_2,
                label=label2,
                marker='o', c='firebrick', alpha=0.2)
    plt.legend(fontsize=14, frameon=True, fancybox=True, markerscale=2.)
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()
    
def plot_resid_hexbin(label_keys, tgt_stellar_labels, pred_stellar_labels,
                      x_label='', y_lims=[1000, 1, 1, 1, 10], 
                      gridsize=(100,50), max_counts=30, cmap='ocean_r', n_std=3,
                      savename=None):
    
    fig, axes = plt.subplots(len(label_keys), 1, 
                             figsize=(10, len(label_keys)*2.5))

    #if not hasattr(axes, 'len'):
    #    axes = [axes]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]
        # Make label pretty
        if label_key=='teff':
            label_key = 'T$_{\mathrm{eff}}$ [K]'
        if label_key=='feh':
            label_key = '[Fe/H]'
        if label_key=='logg':
            label_key = '$\log{g}$'
        if label_key=='alpha':
            label_key = r'[$\alpha$/H]'
        if label_key=='vrad':
            label_key = r'$v_{\mathrm{rad}}$ [km/s]'
            
        # Calculate residual
        diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
        
        # Plot
        tgts = tgt_stellar_labels[:,i]
        x_range = [np.max([np.min(tgts), np.median(tgts)-n_std*np.std(tgts)]),
                   np.min([np.max(tgts), np.median(tgts)+n_std*np.std(tgts)])]
        
        hex_data = ax.hexbin(tgt_stellar_labels[:,i], diff, gridsize=gridsize, cmap=cmap,
                                 extent=(x_range[0], x_range[1], -y_lims[i], y_lims[i]), 
                                 bins=None, vmax=max_counts) 
        
        # Annotate with statistics
        if 'eff' in label_key:
            ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.mean(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        elif 'rad' in label_key:
            ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.mean(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
        else:
            ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)),
                    (0.75,0.8), size=4*len(label_keys), xycoords='axes fraction', 
                    bbox=bbox_props)
            
        # Axis params
        ax.set_xlabel('%s %s' % (x_label, label_key), size=4*len(label_keys))
        ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(label_keys))
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])

        ax.tick_params(labelsize=2.8*len(label_keys))
        ax.grid()
    
    # Colorbar
    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(hex_data, cax=cbar_ax)
    cbar.set_label('Counts', size=4*len(label_keys))
    
    #plt.tight_layout()
        
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()