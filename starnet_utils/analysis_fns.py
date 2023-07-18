import h5py
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_loader import batch_to_device

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines

import seaborn as sns

from string import ascii_lowercase

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

def plot_progress(losses, y_lims=(0,1), x_lim=None,
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize
        
    fig = plt.figure(figsize=(9,3))
    
    gs = gridspec.GridSpec(1, 1)
    
    ax1 = plt.subplot(gs[0])
        
    ax1.set_title('(a) Objective Function', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_source_loss'],
             label=r'Source Train', c='r')
    ax1.plot(losses['batch_iters'], losses['val_source_loss'],
             label=r'Source Val', c='r', ls='--')
    ax1.plot(losses['batch_iters'], losses['val_target_loss'],
             label=r'Target Val', c='k', ls='--')
    ax1.set_ylabel('Loss',fontsize=fontsize)
    
    if x_lim is not None:
        ax1.set_xlim(x_lim[0], x_lim[1])
    else:
        ax1.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
    ax1.set_ylim(*y_lims)
    ax1.set_xlabel('Batch Iterations',fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize_small)
    ax1.grid(True)
    ax1.legend(fontsize=fontsize_small, ncol=1)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()

def dataset_inference(model, dataloader, device):

    # Set parameters to not trainable
    model.eval()
    
    # Estimate labels of the validation spectra
    tgt_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            
            # Switch to GPU if available
            batch = batch_to_device(batch, device)
    
            # Forward propagation (and denormalize outputs)
            label_preds = model(batch['spectrum'], 
                                norm_in=True, 
                                denorm_out=True)
            
            # Save batch data for comparisons
            pred_labels.append(label_preds.cpu().data.numpy())
            tgt_labels.append(batch['labels'].cpu().data.numpy())
            
    pred_labels = np.concatenate(pred_labels)
    tgt_labels = np.concatenate(tgt_labels)

    return tgt_labels, pred_labels

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