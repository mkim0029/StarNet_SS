import torch
import os
import numpy as np
import configparser
import glob

# Directories
cur_dir = os.path.dirname(__file__)
model_dir = os.path.join(cur_dir, 'models/')
config_dir = os.path.join(cur_dir, 'configs/')

models_compare = []
for i in range(4, 55):
    model_name = 'starnet_cnv2_%i'%i

    model_filenames =  glob.glob(os.path.join(model_dir,model_name+'_*.pth.tar'))
    if len(model_filenames)==0:
        model_filenames =  [os.path.join(model_dir,model_name+'.pth.tar')]

    for model_filename in model_filenames:
        if 'lp' in model_filename:
            continue
        
        try:
            # Try loading model
            checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
        except:
            print('Model %i broken. Rerunning' %i)    
            '''
            try:
                os.remove(model_filename)
            except:
                pass

            cmd = 'python launch_starnet_ss.py %s -n 1' % model_name
            print(cmd)
            os.system(cmd)
            '''
            continue

        model_name = model_filename.split('/')[-1].split('.')[0]
            
        # Model configuration
        config = configparser.ConfigParser()
        config.read(config_dir+model_name+'.ini')

        losses = dict(checkpoint['losses'])
        print(losses.keys())
        train_loss = np.mean(losses['train_loss'][-3:])
        val_loss = np.mean(losses['val_loss'][-3:])
        feat_loss = np.mean(losses['val_feats'][-3:])

        models_compare.append([i, train_loss, val_loss, 
                               feat_loss])
        print('Model %i: %s' % (i, config['Notes']['comment']))
        print('\tBatch iters: %i' % (losses['batch_iters'][-1]))
        print('\tTrain Loss: %0.4f' % (train_loss))
        print('\tVal Loss: %0.4f' % (val_loss))
        print('\tFeatures: %0.4f\n' % (feat_loss))

models_compare = np.array(models_compare)
print('Model %i performed the best at predicting training spectra with %0.5f' % (models_compare[np.nanargmin(models_compare[:,1]),0], models_compare[np.nanargmin(models_compare[:,1]),1]))
print('Model %i performed the best at predicting validation spectra with %0.5f' % (models_compare[np.nanargmin(models_compare[:,2]),0],
models_compare[np.nanargmin(models_compare[:,2]),2]))
print('Model %i performed the best at matchin features with %0.5f' % (models_compare[np.nanargmin(models_compare[:,3]),0], 
                            models_compare[np.nanargmin(models_compare[:,3]),3]))
