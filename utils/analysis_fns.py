import numpy as np
from uncertainties import unumpy
import torch

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
    if 'train_src_labels' in losses.keys():
        num_ax += 1
    if 'val_feats' in losses.keys():
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
    if 'train_src_labels' in losses.keys():
        
        ax = plt.subplot(gs[cur_ax])
        axs.append(ax)
    
        ax.set_title('(%s) Stellar Labels' % (ascii_lowercase[cur_ax]), 
                     fontsize=fontsize)
        ax.plot(losses['batch_iters'], np.array(losses['train_src_labels'])[:],
                 label=r'Training', c='r')
        ax.plot(losses['batch_iters'], np.array(losses['val_src_labels'])[:],
                 label=r'Validation$_{src}$', c='grey')
        if 'val_tgt_labels' in losses.keys():
            ax.plot(losses['batch_iters'], np.array(losses['val_tgt_labels'])[:],
                 label=r'Validation$_{tgt}$', c='k')
        ax.set_ylabel('Distance',fontsize=fontsize)
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
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
    
def predict_labels(model, dataset, batchsize=16):
    
    print('Predicting on %i spectra...' % (len(dataset)))
    
    model.eval_mode()
    
    tgt_stellar_labels = []
    pred_stellar_labels = []
    sigma_stellar_labels = []
    tgt_task_labels = []
    pred_task_labels = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for indx in range(len(dataset)):
            batch = dataset.__getitem__(indx)

            # Collect target data
            if len(batch['task labels'])>0:
                tgt_task_labels.append(batch['task labels'].data.numpy())
            tgt_stellar_labels.append(batch['stellar labels'].data.numpy())

            # Perform forward propagation
            if model.network_type.lower()=='cnn':
                model_outputs = model(batch['spectrum chunks'].squeeze(0), 
                                      batch['wave grid chunks'].squeeze(0),
                                      batch['pixel_indx'].squeeze(0),
                                      norm_in=True, denorm_out=True)
                
            elif model.network_type.lower()=='transformer':
                model_outputs_ = []
                for i in range(0, batch['spectrum chunks'].size(0), batchsize):
                    model_outputs_.append(model(batch['spectrum chunks'][i:i+batchsize], 
                                          batch['wave grid chunks'][i:i+batchsize],
                                          batch['pixel_indx'][i:i+batchsize],
                                          norm_in=True, denorm_out=True))
                model_outputs = {}
                for k in model_outputs_[0].keys():
                    model_outputs[k] = torch.cat([model_outputs_[i][k] for i in range(len(model_outputs_))])

            if len(model.tasks)>0:
                pred_task_labels.append(model_outputs['task_labels'].data.numpy())

            if model.est_unc:
                # Weight the average using the variance
                # using error propagation to get the final uncertainties
                stellar_labels_pred_and_unc = unumpy.uarray(model_outputs['labels'].data.numpy(), 
                                                            model_outputs['labels_unc'].data.numpy())
                stellar_labels_sigma = model_outputs['labels_unc'].data.numpy()

                stellar_labels_pred_and_unc = np.sum(1/stellar_labels_sigma**2 * stellar_labels_pred_and_unc, 
                                                             axis=0) / np.sum(1/stellar_labels_sigma**2, 
                                                             axis=0)

                stellar_labels_pred = [stellar_labels_pred_and_unc[i].n for i in range(len(stellar_labels_pred_and_unc))]
                stellar_labels_sigma = [stellar_labels_pred_and_unc[i].s for i in range(len(stellar_labels_pred_and_unc))]

            else:
                # Take average from all spectrum chunk predictions
                pred_stellar_labels.append(torch.mean(model_outputs['labels'], axis=0).data.numpy())

        tgt_stellar_labels = np.vstack(tgt_stellar_labels)
        pred_stellar_labels = np.vstack(pred_stellar_labels)
        if model.est_unc:
            sigma_stellar_labels = np.vstack(sigma_stellar_labels)
        #if len(model.tasks)>0:
            #tgt_task_labels = np.hstack(tgt_task_labels).T
            #pred_task_labels = np.hstack(pred_task_labels).T 
        
    return (tgt_stellar_labels, pred_stellar_labels, 
            sigma_stellar_labels, tgt_task_labels, pred_task_labels)


def predict_ensemble(models, dataset, batchsize=16):
    
    tgt_stellar_labels = []
    pred_stellar_labels = []
    sigma_stellar_labels = []
    tgt_task_labels = []
    pred_task_labels = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for indx in range(len(dataset)):
            batch = dataset.__getitem__(indx)

            # Collect target data
            if len(batch['task labels'])>0:
                tgt_task_labels.append(batch['task labels'].data.numpy())
            tgt_stellar_labels.append(batch['stellar labels'].data.numpy())

            pred_labels = []
            for model in models:

                # Perform forward propagation
                if model.network_type.lower()=='cnn':
                    model_outputs = model(batch['spectrum chunks'].squeeze(0), 
                                          batch['wave grid chunks'].squeeze(0),
                                          batch['pixel_indx'].squeeze(0),
                                          norm_in=True, denorm_out=True)
                elif model.network_type.lower()=='transformer':
                    model_outputs_ = []
                    for i in range(0, batch['spectrum chunks'].size(0), batchsize):
                        model_outputs_.append(model(batch['spectrum chunks'][i:i+batchsize], 
                                              batch['wave grid chunks'][i:i+batchsize],
                                              batch['pixel_indx'][i:i+batchsize],
                                              norm_in=True, denorm_out=True))
                    model_outputs = {}
                    for k in model_outputs_[0].keys():
                        model_outputs[k] = torch.cat([model_outputs_[i][k] for i in range(len(model_outputs_))])

                pred_labels.append(model_outputs['labels'].data.numpy())

            # Calculate mean and std across ensemble of predictions
            stellar_labels_pred = np.array(pred_labels)
            stellar_labels_sigma = np.std(pred_labels,0)
            stellar_labels_pred = np.mean(pred_labels, 0)

            # Weight the average using the variance
            # using error propagation to get the final uncertainties
            stellar_labels_pred_and_unc = unumpy.uarray(stellar_labels_pred, 
                                                        stellar_labels_sigma)

            stellar_labels_pred_and_unc = np.sum(1/stellar_labels_sigma**2 * stellar_labels_pred_and_unc, 
                                                         axis=0) / np.sum(1/stellar_labels_sigma**2, 
                                                         axis=0)

            stellar_labels_pred = [stellar_labels_pred_and_unc[i].n for i in range(len(stellar_labels_pred_and_unc))]
            stellar_labels_sigma = [stellar_labels_pred_and_unc[i].s for i in range(len(stellar_labels_pred_and_unc))]
            
            pred_stellar_labels.append(stellar_labels_pred)
            sigma_stellar_labels.append(stellar_labels_sigma)


        tgt_stellar_labels = np.vstack(tgt_stellar_labels)
        pred_stellar_labels = np.vstack(pred_stellar_labels)
        sigma_stellar_labels = np.vstack(sigma_stellar_labels)
        
    return (tgt_stellar_labels, pred_stellar_labels, 
            sigma_stellar_labels, tgt_task_labels, pred_task_labels)

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
    
def plot_one_to_one(label_keys, tgt_stellar_labels, pred_stellar_labels, est_unc):

    fig, axes = plt.subplots(len(label_keys), 1, figsize=(6, len(label_keys)*6))

    for i, ax in enumerate(axes):
        label_min = np.min(tgt_stellar_labels[:,i])-0.5*np.std(tgt_stellar_labels[:,i])
        label_max = np.max(tgt_stellar_labels[:,i])+0.5*np.std(tgt_stellar_labels[:,i])
        ax.scatter(tgt_stellar_labels[:,i], 
                    pred_stellar_labels[:,i], 
                    alpha=0.5, s=5, zorder=1, c='maroon')
        ax.set_xlabel(r'Target %s' % label_keys[i], size=4*len(label_keys))
        ax.set_ylabel(r'Predicted %s' % label_keys[i], size=4*len(label_keys))
        ax.plot([label_min, label_max], [label_min, label_max],'k--')
        ax.set_xlim(label_min, label_max)
        ax.set_ylim(label_min, label_max)

    plt.tight_layout()
    plt.show()