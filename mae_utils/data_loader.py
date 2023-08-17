import numpy as np
import torch
import h5py

def apply_slope(spectrum, slope_mean, slope_std):
    # Create random slope value
    slope = np.random.normal(slope_mean, slope_std)
    # Add to the spectrum
    spectrum += np.arange(len(spectrum))*slope
    return spectrum, slope

def apply_bias(spectrum, bias_mean, bias_std):
    # Create random bias value
    bias = np.random.normal(bias_mean, bias_std)
    # Add to the spectrum
    spectrum += bias
    return spectrum, bias

def apply_sine(spectrum, amp, period, phi):
    # Add sine wave to the spectrum
    spectrum += amp*np.sin(np.linspace(0,2*np.pi*period, len(spectrum)) + phi)
    return spectrum

def add_noise(x, noise_factor=0.07):

    if type(noise_factor) == float or type(noise_factor) == int or type(noise_factor) == np.float64:
        noise_factor = noise_factor*np.median(x)
        noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x += noise
    else:
        raise ValueError('Noise parameter must be a float or integer')
    return x

def dropout_chunks(spectrum, max_chunks=10, max_chunk_size=200):
    # Number of zero chunks to insert
    num_chunks = np.random.randint(0, max_chunks)
    for i in range(num_chunks):
        # Number of consecutive zeros
        chunk_size = np.random.randint(0, max_chunk_size)
        
        # Starting location of chunk
        chunk_start_indx = np.random.randint(0, len(spectrum)-chunk_size)
        
        # Set flux values to zero
        spectrum[chunk_start_indx:chunk_start_indx+chunk_size] = 0.
    
    return spectrum

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

class SpectraDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the spectral datasets.
    """

    def __init__(self, data_file, dataset, multimodal_keys, unimodal_keys, 
                 continuum_normalize, divide_by_median, label_survey=None,
                 augs=None, aug_means=None, aug_stds=None, median_thresh=0., std_min=0.01,
                 add_noise=False, max_noise_factor=0.1):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.continuum_normalize = continuum_normalize
        self.divide_by_median = divide_by_median
        self.label_survey = label_survey
        self.median_thresh = median_thresh
        self.std_min = std_min
        self.augs = augs
        self.aug_means = aug_means
        self.aug_stds = aug_stds
        self.add_noise = add_noise
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
    
    def apply_augmentations(self, spectrum):

        if self.add_noise:
            # Determine noise factor
            noise_factor = np.random.uniform(0.0001, self.max_noise_factor)
            spectrum = add_noise(spectrum, noise_factor=noise_factor)
                    
        # Perform augmentations according to distributions
        if self.augs is not None:
            sine_aug = False
            for t, tm, ts in zip(self.augs, self.aug_means, self.aug_stds):
                if t.lower()=='slope':
                    spectrum, slope = apply_slope(spectrum, tm, ts)
                if t.lower()=='bias':
                    spectrum, bias = apply_bias(spectrum, tm, ts)
                if t.lower()=='sine amp':
                    sine_amp = np.abs(np.random.normal(tm, ts))
                    sine_aug = True
                if t.lower()=='sine period':
                    sine_period = np.abs(np.random.normal(tm, ts))
                    sine_aug = True
                if t.lower()=='sine phi':
                    sine_phi = np.random.normal(tm, ts)
                    sine_aug = True

            if sine_aug:
                spectrum = apply_sine(spectrum, sine_amp, sine_period, sine_phi)
            
        spectrum = torch.from_numpy(spectrum.astype(np.float32))
            
        return spectrum
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
                
            # Load spectrum
            spectrum = f['spectra %s' % self.dataset][idx]
            spectrum[spectrum<-1] = -1.
            
            data_keys = f.keys()
            # Load target stellar labels for classifiers
            multimodal_labels = []
            for k in self.multimodal_keys:
                data_key = k + ' %s' % self.dataset
                if self.label_survey is not None:
                    data_key = self.label_survey + ' ' + data_key
                if data_key in data_keys:
                    multimodal_labels.append(f[data_key][idx])
                elif ('mg' in data_key) & (data_key.replace('mg', 'alpha') in data_keys):
                    multimodal_labels.append(f[data_key.replace('mg', 'alpha')][idx])
                else:
                    multimodal_labels.append(np.nan)
            multimodal_labels = torch.from_numpy(np.asarray(multimodal_labels).astype(np.float32))
            
            # Load target stellar labels for linear predictors
            unimodal_labels = []
            for k in self.unimodal_keys:
                data_key = k + ' %s' % self.dataset
                if data_key in data_keys:
                    unimodal_labels.append(f[data_key][idx])
                elif ('mg' in data_key) & ('alpha %s' % self.dataset in data_keys):
                    unimodal_labels.append(f['alpha %s' % self.dataset][idx])
                else:
                    unimodal_labels.append(np.nan)
            unimodal_labels = torch.from_numpy(np.asarray(unimodal_labels).astype(np.float32))
            
            if self.continuum_normalize:
                # Divide spectrum by its estimated continuum
                spectrum = spectrum/f['continua %s' % self.dataset][idx]
            
        if self.divide_by_median:
            # Divide spectrum by its median to centre it around 1
            spectrum = spectrum/np.median(spectrum[spectrum>self.median_thresh])

        # Apply augmentations to entire spectrum as well
        spectrum = self.apply_augmentations(spectrum)

        return {'spectrum':spectrum,
                'multimodal labels':multimodal_labels,
                'unimodal labels':unimodal_labels}