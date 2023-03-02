import numpy as np
from uncertainties import unumpy
import torch

import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_loader import batch_to_device

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
from string import ascii_lowercase

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

def plot_progress(losses, tasks, y_lims=[(0,1)], x_lim=None, 
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = 1+len(tasks)
    if 'train_src_mm_labels' in losses.keys():
        num_ax += 1
    if 'train_src_um_labels' in losses.keys():
        num_ax += 1
    if 'val_feats' in losses.keys():
        num_ax += 1
    if 'train_src_feats' in losses.keys():
        num_ax += 1
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    ax1 = plt.subplot(gs[0])
    
    axs = [ax1]
    
    ax1.set_title('(a) Objective Function', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_loss'],
             label=r'Training', c='r')
    '''ax1.plot(losses['batch_iters'], losses['val_loss'],
             label=r'Validation', c='k')'''
    ax1.set_ylabel('Loss',fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    ax1.legend(fontsize=fontsize_small)
    
    cur_ax = 1
    if 'train_src_mm_labels' in losses.keys():
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
    
        ax.set_title('(%s) Multimodal Labels' % (ascii_lowercase[cur_ax]), 
                     fontsize=fontsize)
        ax.plot(losses['batch_iters'], np.array(losses['train_src_mm_labels'])[:],
                 label=r'Training', c='r')
        ax.set_ylabel('NLL',fontsize=fontsize)
        ax.set_ylim(*y_lims[cur_ax])
        ax.legend(fontsize=fontsize_small, ncol=2)
        cur_ax+=1
    if 'train_src_um_labels' in losses.keys():
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
    
        ax.set_title('(%s) Unimodal Labels' % (ascii_lowercase[cur_ax]), 
                     fontsize=fontsize)
        ax.plot(losses['batch_iters'], np.array(losses['train_src_um_labels'])[:],
                 label=r'Training', c='r')
        ax.set_ylabel('MSE',fontsize=fontsize)
        ax.set_ylim(*y_lims[cur_ax])
        ax.legend(fontsize=fontsize_small, ncol=2)
        cur_ax+=1
        
    if 'val_feats' in losses.keys():
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
    
        ax.set_title('(%s) Feature Matching Score' % (ascii_lowercase[cur_ax]), 
                     fontsize=fontsize)
        ax.plot(losses['batch_iters'], np.array(losses['val_feats'])[:],
                 label=r'Validation', c='k')
        ax.set_ylabel('Distance',fontsize=fontsize)
        ax.set_ylim(*y_lims[cur_ax])
        ax.legend(fontsize=fontsize_small, ncol=1)
        cur_ax+=1
        
    if 'train_src_feats' in losses.keys():
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
    
        ax.set_title('(%s) Feature Loss' % (ascii_lowercase[cur_ax]), 
                     fontsize=fontsize)
        ax.plot(losses['batch_iters'], np.array(losses['train_src_feats'])[:],
                 label=r'Training$_{src}$', c='r')
        ax.plot(losses['batch_iters'], np.array(losses['train_tgt_feats'])[:],
                 label=r'Training$_{tgt}$', c='k')
        ax.set_ylabel('Distance',fontsize=fontsize)
        ax.set_ylim(*y_lims[cur_ax])
        ax.legend(fontsize=fontsize_small, ncol=1)
        cur_ax+=1
    
    for i, task in enumerate(tasks):
        
        train_src_task_losses = np.array(losses['train_src_tasks'])[:,i]        
        train_tgt_task_losses = np.array(losses['train_tgt_tasks'])[:,i]
        #val_src_task_losses = np.array(losses['val_src_tasks'])[:,i]
        #val_tgt_task_losses = np.array(losses['val_tgt_tasks'])[:,i]
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
        
        ax.plot(losses['batch_iters'], train_src_task_losses,
                 label=r'Training$_{src}$', c='r')
        ax.plot(losses['batch_iters'], train_tgt_task_losses,
                 label=r'Training$_{tgt}$', c='maroon')
        '''ax.plot(losses['batch_iters'], val_src_task_losses,
                 label=r'Validation$_{src}$', c='grey')
        ax.plot(losses['batch_iters'], val_tgt_task_losses,
                 label=r'Validation$_{tgt}$', c='k')'''
        
        ax.set_title('(%s) %s Task' % (ascii_lowercase[cur_ax], task.capitalize()), 
                     fontsize=fontsize)
        ax.set_ylabel('Distance',fontsize=fontsize)
        ax.set_ylim(*y_lims[cur_ax])
        ax.legend(fontsize=fontsize_small, ncol=2)
        cur_ax+=1
        
    
    for i, ax in enumerate(axs):
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
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
        ax.plot(losses['batch_iters'], losses['val_src_%s' % key],
                 label=r'Source', c='k')
        ax.plot(losses['batch_iters'], losses['val_tgt_%s' % key],
                 label=r'Target', c='r')
        ax.set_ylabel('NLL',fontsize=fontsize)
        ax.set_ylim(*y_lims[i])
        ax.legend(fontsize=fontsize_small, ncol=2)

        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_label_MAE(losses, label_keys, y_lims=[(0,1)], x_lim=None, 
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = len(label_keys)
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    for i, key in enumerate(label_keys):
        
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
        
        ax = plt.subplot(gs[i])
    
        ax.set_title('(a) %s' % (label_key), fontsize=fontsize)
        ax.plot(losses['batch_iters'], losses['val_src_%s' % key],
                 label=r'Source', c='k')
        ax.plot(losses['batch_iters'], losses['val_tgt_%s' % key],
                 label=r'Target', c='r')
        ax.set_ylabel('MAE',fontsize=fontsize)
        ax.set_ylim(*y_lims[i])
        ax.legend(fontsize=fontsize_small)
        
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
    
def predict_labels(model, dataset, device, batchsize=16, take_mode=False, 
                   combine_batch_probs=False, take_batch_mode=False,
                   chunk_indices=None, chunk_weights=None):
    
    print('Predicting on %i spectra...' % (len(dataset)))
    try:
        model.eval_mode()
    except AttributeError:
        model.module.eval_mode()
    
    tgt_mm_labels = []
    tgt_um_labels = []
    pred_mm_labels = []
    pred_um_labels = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for indx in range(len(dataset)):
            batch = dataset.__getitem__(indx)
            batch = batch_to_device(batch, device)

            # Collect target data
            tgt_mm_labels.append(batch['multimodal labels'].data.cpu().numpy())
            if len(batch['unimodal labels'])>0:
                tgt_um_labels.append(batch['unimodal labels'].data.cpu().numpy())

            try:
                # Perform forward propagation
                model_outputs = model(batch['spectrum chunks'].squeeze(0), 
                                      batch['pixel_indx'].squeeze(0),
                                      norm_in=True, denorm_out=True,
                                      take_mode=take_mode,
                                      combine_batch_probs=combine_batch_probs,
                                      take_batch_mode=take_batch_mode,
                                      chunk_indices=chunk_indices.to(device), 
                                      chunk_weights=chunk_weights.to(device))
            except AttributeError:
                model_outputs = model.module(batch['spectrum chunks'].squeeze(0), 
                                             batch['pixel_indx'].squeeze(0),
                                             norm_in=True, denorm_out=True,
                                             take_mode=take_mode,
                                             combine_batch_probs=combine_batch_probs,
                                             take_batch_mode=take_batch_mode,
                                             chunk_indices=chunk_indices.to(device), 
                                             chunk_weights=chunk_weights.to(device))

            # Take average from all spectrum chunk predictions
            pred_mm_labels.append(np.mean(model_outputs['multimodal labels'].data.cpu().numpy(), axis=0))
            if len(batch['unimodal labels'])>0:
                pred_um_labels.append(np.mean(model_outputs['unimodal labels'].data.cpu().numpy(), axis=0))

        tgt_mm_labels = np.vstack(tgt_mm_labels)
        pred_mm_labels = np.vstack(pred_mm_labels)
        if len(tgt_um_labels)>0:
            tgt_um_labels = np.vstack(tgt_um_labels)
            pred_um_labels = np.vstack(pred_um_labels)

        
    return tgt_mm_labels, tgt_um_labels, pred_mm_labels, pred_um_labels

def predict_ensemble(models, dataset, channel_starts = [0, 11880, 25880], batchsize=16, take_mode=False):
    
    # Create a list of the starting indices of the chunks
    channel_starts = [0, 11880, 25880]
    starting_indices = []
    for cs, i in zip(channel_starts, dataset.starting_indices):
        starting_indices.append(cs+i)
    starting_indices = np.concatenate(starting_indices)
    
    tgt_stellar_labels = []
    pred_stellar_labels = []
    sigma_stellar_labels = []
    tgt_task_labels = []
    pred_task_labels = []
    
    chunk_sigmas = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for indx in range(len(dataset)):
            batch = dataset.__getitem__(indx)

            # Collect target data
            if len(batch['task labels'])>0:
                tgt_task_labels.append(batch['task labels'].data.numpy())
            tgt_stellar_labels.append(np.concatenate((batch['multimodal labels'].data.numpy(),
                                                      batch['unimodal labels'].data.numpy())))

            pred_labels = []
            for model in models:

                # Perform forward propagation
                model_outputs = model(batch['spectrum chunks'].squeeze(0), 
                                      batch['pixel_indx'].squeeze(0),
                                      norm_in=True, denorm_out=True,
                                     take_mode=take_mode)

                sl = np.hstack((model_outputs['multimodal labels'].data.numpy(), 
                            model_outputs['unimodal labels'].data.numpy()))
                
                pred_labels.append(sl)

            # Calculate mean and std across ensemble of predictions
            stellar_labels_pred = np.array(pred_labels)
            stellar_labels_sigma = np.std(pred_labels,0)
            stellar_labels_pred = np.mean(pred_labels, 0)
            
            # Store the sigmas for each chunk
            batch_pix_indices = batch['pixel_indx'].squeeze(0).squeeze(1).data.numpy()
            num_labels = stellar_labels_pred.shape[1]
            sigs = []
            for pix_index in starting_indices:
                
                chunk_index = np.where(batch_pix_indices == pix_index)[0]
                if len(chunk_index)==0:
                    sigs.append(np.empty((num_labels,)) * np.nan)
                else:
                    sigs.append(stellar_labels_sigma[chunk_index][0])
            chunk_sigmas.append(sigs)

            # Weight the average using the variance
            # using error propagation to get the final uncertainties
            stellar_labels_pred_and_unc = unumpy.uarray(stellar_labels_pred, 
                                                        stellar_labels_sigma)

            stellar_labels_pred_and_unc = np.sum(1/(stellar_labels_sigma + 1e-5)**2 * stellar_labels_pred_and_unc,
                                                 axis=0) / np.sum(1/(stellar_labels_sigma + 1e-5)**2, axis=0)

            stellar_labels_pred = [stellar_labels_pred_and_unc[i].n for i in range(len(stellar_labels_pred_and_unc))]
            stellar_labels_sigma = [stellar_labels_pred_and_unc[i].s for i in range(len(stellar_labels_pred_and_unc))]
            
            pred_stellar_labels.append(stellar_labels_pred)
            sigma_stellar_labels.append(stellar_labels_sigma)

        tgt_stellar_labels = np.vstack(tgt_stellar_labels)
        pred_stellar_labels = np.vstack(pred_stellar_labels)
        sigma_stellar_labels = np.vstack(sigma_stellar_labels)
        chunk_sigmas = np.array(chunk_sigmas)
        
        chunk_sigmas = np.nanmean(chunk_sigmas, axis=0).T
        
    return (tgt_stellar_labels, pred_stellar_labels, 
            sigma_stellar_labels, tgt_task_labels, pred_task_labels, chunk_sigmas, starting_indices)

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

def plot_resid(label_keys, tgt_stellar_labels, pred_stellar_labels, sigma_stellar_labels=None,
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
        ax.set_xlabel('%s' % (label_key), size=4*len(label_keys))
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
        plt.savefig(savename, facecolor='white', transparent=False, dpi=200,
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
    
def plot_resid_boxplot(label_keys, tgt_stellar_labels, pred_stellar_labels, 
                       sigma_stellar_labels=None,
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
            
            box_positions = []
            box_data = []
            for tgt_val in np.unique(tgt_stellar_labels[:,i]):
                indices = np.where(tgt_stellar_labels[:,i]==tgt_val)[0]
                if len(indices)>2:
                    box_positions.append(tgt_val)
                    box_data.append(diff[indices])
            box_width = np.mean(np.diff(box_positions))/2
            
            ax.boxplot(box_data, positions=box_positions, widths=box_width)
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
        ax.set_xlabel('%s' % (label_key), size=4*len(label_keys))
        ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(label_keys))
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_xlim(np.min(box_positions)-box_width*2, np.max(box_positions)+box_width*2)
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])
        
        if 'eff' in label_key:
            ax.set_xticks(box_positions)
            ax.set_xticklabels(np.array(box_positions).astype(int))

        ax.tick_params(labelsize=3*len(label_keys))
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
    
def plot_wave_sigma(chunk_sigmas, label_keys, wave_grid_file, starting_indices, num_fluxes,
                    y_lims=[(0,1)], fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    wave_grid = np.load(wave_grid_file)
    
    num_ax = len(label_keys)
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    for i, key in enumerate(label_keys):
        
        
        s_vals = chunk_sigmas[i]
        wave_sigma = np.empty((len(s_vals), len(wave_grid))) * np.nan
        for j, indx in enumerate(starting_indices):
            wave_sigma[j, indx:indx+num_fluxes] = s_vals[j]
        wave_sigma = np.nanmean(wave_sigma, axis=0)
        
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
                
        ax = plt.subplot(gs[i])
    
        ax.set_title('(%s) %s' % (ascii_lowercase[i],label_key), fontsize=fontsize)
        ax.plot(wave_grid, wave_sigma, c='r')
        ax.set_ylabel('$\sigma$',fontsize=fontsize)
        ax.set_ylim(*y_lims[i])
        ax.set_xlabel('Wavelength ($\AA$)',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=200,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def save_results(pred_stellar_labels_train, pred_stellar_labels_val, 
                 datafile, label_keys, savename):

    pred_labels = np.vstack((pred_stellar_labels_train, pred_stellar_labels_val))
    
    # Load identifiers
    with h5py.File(datafile, 'r') as f:
        cname = np.concatenate((f['cname train'][:], f['cname val'][:]))
        targid = np.concatenate((f['targid train'][:], f['targid val'][:]))
        wprov = np.concatenate((f['wprov train'][:], f['wprov val'][:]))
        
    with h5py.File(savename, 'w') as f:
        # Save labels
        for i, key in enumerate(label_keys):
            f.create_dataset(key, data=pred_labels[:,i])
            
        f.create_dataset('cname', data=cname)
        f.create_dataset('targid', data=targid)
        f.create_dataset('wprov', data=wprov)