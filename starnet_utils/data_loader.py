import numpy as np
import torch
import h5py

def add_noise(x, noise_factor=0.07):

    if type(noise_factor) == float or type(noise_factor) == int or type(noise_factor) == np.float64:
        noise_factor = noise_factor*np.median(x)
        noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x += noise
    else:
        raise ValueError('Noise parameter must be a float or integer')
    return x

def batch_to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], list):
            for i in range(len(batch[k])):
                batch[k][i] = batch[k][i].to(device)
        else:
            try:
                batch[k] = batch[k].to(device)
            except AttributeError:
                batch[k] = torch.tensor(batch[k]).to(device)
    return batch

class SpectraDatasetSimple(torch.utils.data.Dataset):
    
    """
    Dataset loader for the spectral datasets.
    """

    def __init__(self, data_file, dataset, label_keys, max_noise_factor=0.0):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.label_keys = label_keys
        self.max_noise_factor = max_noise_factor
        # Determine the number of pixels in each spectrum
        self.num_pixels = self.determine_num_pixels()
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_spectra = len(f['spectra %s' % self.dataset])
        return num_spectra
    
    def determine_num_pixels(self):
        with h5py.File(self.data_file, "r") as f:    
            num_pixels = f['spectra %s' % self.dataset].shape[1]
        return num_pixels
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
                
            # Load spectrum
            spectrum = f['spectra %s' % self.dataset][idx]
            spectrum[spectrum<-1] = -1.

            # Add random noise
            if self.max_noise_factor>0.0:
                # Determine noise factor
                noise_factor = np.random.uniform(0.0001, self.max_noise_factor)
                spectrum = add_noise(spectrum, noise_factor=noise_factor)
            
            spectrum = torch.from_numpy(spectrum.astype(np.float32))
            
            # Load target stellar labels
            data_keys = f.keys()
            labels = []
            for k in self.label_keys:
                data_key = k + ' %s' % self.dataset
                if data_key in data_keys:
                    labels.append(f[data_key][idx])
                else:
                    labels.append(np.nan)
            labels = torch.from_numpy(np.asarray(labels).astype(np.float32))
            
        # Return full spectrum and target labels
        return {'spectrum':spectrum,
                'labels':labels}