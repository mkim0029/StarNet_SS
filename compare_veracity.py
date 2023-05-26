import os
import numpy as np
import configparser
import h5py

isochrone_fn = 'data/isochrone_data.h5'

def zscore_norm(data, data_mean, data_std):
    return (data - data_mean) / data_std

def isochrone_comparison(isochrone_fn, teff_vals, logg_vals, feh_vals,
                         label_means=[5300, 3.0, -1.7], label_stds=[1500, 1.5, 1.7]):
    
    '''Compute the distance (in 3D) from each value 
    and its closest point on the isochrone lines.'''
    
    # Load isochrone data
    with h5py.File(isochrone_fn, 'r') as f:
        iso_teff = f['teff'][:]
        iso_logg = f['logg'][:]
        iso_feh = f['feh'][:]
        
    # Calculate difference in the three params after performing z-score normalization
    sq_diffs = []
    for vals, iso_vals, l_m, l_s in zip([teff_vals, logg_vals, feh_vals],
                                        [iso_teff, iso_logg, iso_feh],
                                        label_means, label_stds):
        # Normalize
        vals = zscore_norm(vals, l_m, l_s)
        iso_vals = zscore_norm(iso_vals, l_m, l_s)
        
        # Compute square difference between all values
        sq_diffs.append( (iso_vals.reshape((len(iso_vals),1)) - 
                          vals.reshape((1, len(vals)))) **2 )
    
    # Compute Euclidean distance
    iso_diffs = np.sqrt(np.sum(sq_diffs, axis=0))
        
    # Sample-wise minimum distance
    iso_diffs = np.min(iso_diffs, axis=0)
        
    return iso_diffs

# Directories
cur_dir = os.path.dirname(__file__)
results_dir = os.path.join(cur_dir, 'results/')
config_dir = os.path.join(cur_dir, 'configs/')

models_compare = []
for i in range(290,291):
    model_name = 'starnet_ss_%i'%i        
    try:
        # Try loading results
        target_mm_preds = np.load(os.path.join(results_dir,'%s_target_mm_preds.npy'%model_name)) 
    except:
        print('Model %i not finished I guess...' %i)    
        continue
            
    # Compare predictions against isochrones
    pred_iso_diffs = isochrone_comparison(isochrone_fn, 
                                              target_mm_preds[:3000,0], 
                                              target_mm_preds[:3000,2],
                                              target_mm_preds[:3000,1])
        
    avg_distance = np.mean(pred_iso_diffs)
        
    # Model configuration
    config = configparser.ConfigParser()
    config.read(os.path.join(config_dir, model_name+'.ini'))

    models_compare.append([i, avg_distance])
    print('Model %i: %s' % (i, config['Notes']['comment']))
    print('\tIsochrone Distance: %0.4f' % (avg_distance))

models_compare = np.array(models_compare)
print('Model %i performed the best with a distance of %0.5f' % (models_compare[np.nanargmin(models_compare[:,1]),0], models_compare[np.nanargmin(models_compare[:,1]),1]))