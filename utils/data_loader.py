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

def calc_snr(spectrum):
    # Calculate snr
    n = len(spectrum)
    
    if n>10:
        signal = np.median(spectrum)
        noise  = 0.6052697 * np.median(np.abs(2.0 * spectrum[2:n-2] - spectrum[0:n-4] - spectrum[4:n]))
    else:
        signal = 1
        noise = 1
    return signal/(noise+1e-3)

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

    def __init__(self, data_file, dataset, wave_grid_file, multimodal_keys, unimodal_keys, 
                 continuum_normalize, divide_by_median, chunk_size=None,
                 tasks=None, task_means=None, task_stds=None, median_thresh=0., std_min=0.01,
                 apply_dropout=False, add_noise=False, max_noise_factor=0.1, 
                 random_chunk=False, overlap=0.5, channel_indices=[0,11880,25880],
                 inference_mode=False):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.continuum_normalize = continuum_normalize
        self.divide_by_median = divide_by_median
        self.median_thresh = median_thresh
        if chunk_size is None:
            self.chunk_size = self.determine_num_pixels()
        else:
            self.chunk_size = chunk_size
        self.std_min = std_min
        self.wave_grid = np.load(wave_grid_file).astype(np.float32)
        self.tasks = tasks
        self.task_means = task_means
        self.task_stds = task_stds
        self.apply_dropout = apply_dropout
        self.add_noise = add_noise
        self.max_noise_factor = max_noise_factor
        self.random_chunk = random_chunk
        self.overlap = overlap
        self.channel_indices = channel_indices
        self.inference_mode = inference_mode
        
        # Determine the number of pixels in each spectrum
        self.num_pixels = self.determine_num_pixels()
        
        # Determine starting pixel indices to choose chunks from
        self.starting_indices = self.determine_starting_indices()
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_spectra = len(f['spectra %s' % self.dataset])
        return num_spectra
    
    def determine_starting_indices(self):

        channel_starts = self.channel_indices
        channel_ends = channel_starts[1:] + [self.num_pixels]
        
        
        # Only grab chunks that are entirely within a single channel
        starting_indices = []
        for start_i, end_i in zip(channel_starts, channel_ends):
                starting_indices.append(np.arange(0,
                                                  end_i-start_i-self.chunk_size,
                                                  self.chunk_size*(1-self.overlap)).astype(int))
        return starting_indices
    
    def determine_num_pixels(self):
        with h5py.File(self.data_file, "r") as f:    
            num_pixels = f['spectra %s' % self.dataset].shape[1]
        return num_pixels
    
    def select_random_chunk(self, spectrum, wave_grid, starting_indices, pixel_indx):
        
        # Select one channel
        if len(spectrum)>1:
            channel_num = np.random.randint(len(spectrum))
        else:
            channel_num = 0
        wave_grid = wave_grid[channel_num]
        starting_indices = starting_indices[channel_num]
        spectrum = np.copy(spectrum[channel_num])
        pixel_indx = pixel_indx[channel_num]

        if self.random_chunk:
            # Select random chunk from this channel
            start_indx = np.random.choice(starting_indices)
        else:
            start_indx = 0
        pixel_indx += start_indx

        # Select chunk of the spectrum
        spectrum = spectrum[start_indx:start_indx+self.chunk_size]
        wave_grid = wave_grid[start_indx:start_indx+self.chunk_size]
        
        # Determine centre of the wavelength range
        centre_wave = np.median(wave_grid)

        return spectrum, centre_wave, pixel_indx
    
    def apply_augmentations(self, spectrum, centre_wave):

        if self.add_noise:
            # Determine noise factor
            noise_factor = np.random.uniform(0.0001, self.max_noise_factor)
            spectrum = add_noise(spectrum, noise_factor=noise_factor)
                    
        # Perform augmentations according to tasks
        sine_aug = False
        task_labels = []
        for t, tm, ts in zip(self.tasks, self.task_means, self.task_stds):
            if t.lower()=='wavelength':
                task_labels.append(centre_wave) 
            if t.lower()=='slope':
                spectrum, slope = apply_slope(spectrum, tm, ts)
                task_labels.append(slope)
            if t.lower()=='bias':
                spectrum, bias = apply_bias(spectrum, tm, ts)
                task_labels.append(bias)
            if t.lower()=='snr':
                snr = calc_snr(spectrum[spectrum>0.1])
                task_labels.append(snr)
            if t.lower()=='sine amp':
                sine_amp = np.abs(np.random.normal(tm, ts))
                task_labels.append(sine_amp)
                sine_aug = True
            if t.lower()=='sine period':
                sine_period = np.abs(np.random.normal(tm, ts))
                task_labels.append(sine_period)
                sine_aug = True
            if t.lower()=='sine phi':
                sine_phi = np.random.normal(tm, ts)
                task_labels.append(sine_phi)
                sine_aug = True
            
        if sine_aug:
            spectrum = apply_sine(spectrum, sine_amp, sine_period, sine_phi)
                
        if self.apply_dropout:
            # Dropout random chunks of the spectrum
            spectrum = dropout_chunks(spectrum, 
                                      max_chunks=10, 
                                      max_chunk_size=200)
            
        task_labels = torch.from_numpy(np.array(task_labels).astype(np.float32))
        spectrum = torch.from_numpy(spectrum.astype(np.float32))
            
        return spectrum, task_labels
    
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
                if data_key in data_keys:
                    multimodal_labels.append(f[data_key][idx])
                elif ('mg' in data_key) & ('alpha %s' % self.dataset in data_keys):
                    multimodal_labels.append(f['alpha %s' % self.dataset][idx])
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
        
        if self.inference_mode:
            # Return full spectrum and target labels without applying augmentations
            return {'spectrum':torch.from_numpy(spectrum.astype(np.float32)),
                    'multimodal labels':multimodal_labels,
                    'unimodal labels':unimodal_labels,
                    'spectrum index': 0}
        
        else:
            # Select a random chunk and apply augmentations
            
            # Index of leftmost pixel in each channel
            pixel_indx = self.channel_indices

            # Split spectrum into channels
            wave_grid = []
            spectrum_ = []
            for i in range(len(pixel_indx)):
                if i==(len(pixel_indx)-1):
                    wave_grid.append(self.wave_grid[pixel_indx[i]:])
                    spectrum_.append(spectrum[pixel_indx[i]:])
                else:
                    wave_grid.append(self.wave_grid[pixel_indx[i]:pixel_indx[i+1]])
                    spectrum_.append(spectrum[pixel_indx[i]:pixel_indx[i+1]])
            spectrum = spectrum_

            # Remove channels without info
            wave_grid = [wave_grid[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
            starting_indices = [self.starting_indices[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
            pixel_indx = [pixel_indx[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
            spectrum = [spec for spec in spectrum if np.std(spec)>self.std_min]

            # Select random chunk in the spectrum
            spectrum_chunk, centre_wave, chunk_indx = self.select_random_chunk(spectrum, 
                                                                            wave_grid, 
                                                                            starting_indices, 
                                                                            pixel_indx)

            # Apply augmentations and create array of task labels
            spectrum_chunk, task_labels_chunk = self.apply_augmentations(spectrum_chunk, centre_wave)

            # Apply augmentations to entire spectrum as well
            spectrum, task_labels_full = self.apply_augmentations(np.concatenate(spectrum), 
                                                                  np.mean(wave_grid))

            return {'spectrum':spectrum,
                    'spectrum chunk':spectrum_chunk, 
                    'multimodal labels':multimodal_labels,
                    'unimodal labels':unimodal_labels,
                    'task labels full':task_labels_full,
                    'task labels chunk':task_labels_chunk,
                    'spectrum index': 0,
                    'chunk index':chunk_indx}
