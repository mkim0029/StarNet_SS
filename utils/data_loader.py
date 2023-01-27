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
            batch[k] = batch[k].to(device)
    return batch
    
class WeaveSpectraDataset(torch.utils.data.Dataset):
    
    """
    Dataset for the WEAVE spectral datasets.        
    """

    def __init__(self, data_file, dataset, wave_grid_file, multimodal_keys, unimodal_keys, 
                 continuum_normalize, divide_by_median, num_fluxes,
                 tasks, task_means, task_stds, calc_mutlimodal_vals=False,
                 median_thresh=0., std_min=0.01,
                 apply_dropout=False, add_noise=False, max_noise_factor=0.1, 
                 random_chunk=False, overlap=0.5, load_second_chunk=False):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.continuum_normalize = continuum_normalize
        self.divide_by_median = divide_by_median
        self.median_thresh = median_thresh
        self.num_fluxes = num_fluxes
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
        self.load_second_chunk = load_second_chunk
        
        # Determine starting pixel indices to choose chunks from
        self.starting_indices = self.determine_starting_indices()
        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_spectra = len(f['spectra %s' % self.dataset])
        return num_spectra
    
    def determine_starting_indices(self, 
                                   channel_starts=[0, 11880, 25880],
                                   channel_ends=[11880, 25880, 43480]):

        # Only grab chunks that are entirely within a single channel
        starting_indices = []
        for start_i, end_i in zip(channel_starts, channel_ends):
                starting_indices.append(np.arange(0,
                                                  end_i-start_i-self.num_fluxes,
                                                  self.num_fluxes*(1-self.overlap)).astype(int))
        return starting_indices
    
    def select_random_chunk(self, spectrum, wave_grid, starting_indices, pixel_indx):
        
        # Select one channel
        channel_num = np.random.randint(len(spectrum))
        wave_grid = wave_grid[channel_num]
        starting_indices = starting_indices[channel_num]
        spectrum = spectrum[channel_num]
        pixel_indx = pixel_indx[channel_num]

        if self.random_chunk:
            # Select random chunk from this channel
            start_indx = np.random.choice(starting_indices)
        else:
            start_indx = 0
        pixel_indx += start_indx

        # Select chunk of the spectrum
        spectrum = spectrum[start_indx:start_indx+self.num_fluxes]
        wave_grid = wave_grid[start_indx:start_indx+self.num_fluxes]
        
        # Determine centre of the wavelength range
        centre_wave = np.median(wave_grid)

        return spectrum, centre_wave, pixel_indx
    
    def apply_augmentations(self, spectrum, centre_wave):

        if self.add_noise:
            # Determine noise factor
            noise_factor = np.random.uniform(0.0001, self.max_noise_factor)
            spectrum = add_noise(spectrum, noise_factor=noise_factor)
            
        # Perform augmentations according to tasks
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
            
            # Load stellar labels
            multimodal_labels = np.asarray([f[k + ' %s' % self.dataset][idx] for k in self.multimodal_keys])
            multimodal_labels = torch.from_numpy(multimodal_labels.astype(np.float32))
            unimodal_labels = np.asarray([f[k + ' %s' % self.dataset][idx] for k in self.unimodal_keys])
            unimodal_labels = torch.from_numpy(unimodal_labels.astype(np.float32))
            
            if self.continuum_normalize:
                # Divide spectrum by its estimated continuum
                spectrum = spectrum/f['continua %s' % self.dataset][idx]
            
        if self.divide_by_median:
            # Divide spectrum by its median to centre it around 1
            spectrum = spectrum/np.median(spectrum[spectrum>self.median_thresh])
            
        # Index of leftmost pixel in each channel
        pixel_indx = [0, 11880, 25880]

        # Split spectrum into channels
        wave_grid = [self.wave_grid[pixel_indx[0]:pixel_indx[1]], 
                     self.wave_grid[pixel_indx[1]:pixel_indx[2]], 
                     self.wave_grid[pixel_indx[2]:]]
        spectrum = [spectrum[pixel_indx[0]:pixel_indx[1]], 
                    spectrum[pixel_indx[1]:pixel_indx[2]], 
                    spectrum[pixel_indx[2]:]]

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
        spectrum_chunk, task_labels = self.apply_augmentations(spectrum_chunk, centre_wave)


        if self.load_second_chunk:
            # Select random chunk in the spectrum
            spectrum_chunk2, centre_wave, chunk_indx2 = self.select_random_chunk(spectrum, 
                                                                  wave_grid, 
                                                                  starting_indices, 
                                                                  pixel_indx)

            # Apply augmentations and create array of task labels
            spectrum_chunk2, _ = self.apply_augmentations(spectrum_chunk2, centre_wave)

            return {'spectrum':spectrum_chunk,
                    'spectrum2':spectrum_chunk2,
                    'multimodal labels':multimodal_labels,
                    'unimodal labels':unimodal_labels,
                    'task labels':task_labels,
                    'pixel_indx':chunk_indx,
                    'pixel_indx2':chunk_indx2}

        else:
            return {'spectrum':spectrum_chunk, 
                    'multimodal labels':multimodal_labels,
                    'unimodal labels':unimodal_labels,
                    'task labels':task_labels,
                    'pixel_indx':chunk_indx}
        
class WeaveSpectraDatasetInference(torch.utils.data.Dataset):
    
    """
            
    """

    def __init__(self, data_file, dataset, wave_grid_file, multimodal_keys, unimodal_keys, 
                 continuum_normalize, divide_by_median, num_fluxes,
                 tasks, task_means, task_stds, median_thresh=0., std_min=0.01, 
                 random_chunk=False, overlap=0.5):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.multimodal_keys = multimodal_keys
        self.unimodal_keys = unimodal_keys
        self.continuum_normalize = continuum_normalize
        self.divide_by_median = divide_by_median
        self.median_thresh = median_thresh
        self.num_fluxes = num_fluxes
        self.std_min = std_min
        self.wave_grid = np.load(wave_grid_file).astype(np.float32)
        self.tasks = tasks
        self.task_means = task_means
        self.task_stds = task_stds
        self.overlap = overlap
        self.random_chunk = random_chunk
        
        # Determine starting indices to choose chunks from
        self.starting_indices = self.determine_starting_indices()
        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_spectra = len(f['spectra %s' % self.dataset])
        return num_spectra
    
    def determine_starting_indices(self, 
                                   channel_starts=[0, 11880, 25880],
                                   channel_ends=[11880, 25880, 43480]):

        # Only grab chunks that are entirely within a single channel
        starting_indices = []
        for start_i, end_i in zip(channel_starts, channel_ends):
                starting_indices.append(np.arange(0,
                                                  end_i-start_i-self.num_fluxes,
                                                  self.num_fluxes*self.overlap).astype(int))
        return starting_indices
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
                
            # Load spectrum
            spectrum = f['spectra %s' % self.dataset][idx]
            spectrum[spectrum<-1] = -1.
            
            # Load stellar labels
            multimodal_labels = np.asarray([f[k + ' %s' % self.dataset][idx] for k in self.multimodal_keys])
            multimodal_labels = torch.from_numpy(multimodal_labels.astype(np.float32))
            unimodal_labels = np.asarray([f[k + ' %s' % self.dataset][idx] for k in self.unimodal_keys])
            unimodal_labels = torch.from_numpy(unimodal_labels.astype(np.float32))
            
            if self.continuum_normalize:
                # Divide spectrum by its estimated continuum
                spectrum = spectrum/f['continua %s' % self.dataset][idx]
            
        if self.divide_by_median:
            # Divide spectrum by its median to centre it around 1
            spectrum = spectrum/np.median(spectrum[spectrum>self.median_thresh])

        # Index of leftmost pixel in spectrum
        pixel_indx = [0, 11880, 25880]

        # Split spectrum into channels
        wave_grid = [self.wave_grid[pixel_indx[0]:pixel_indx[1]], 
                     self.wave_grid[pixel_indx[1]:pixel_indx[2]], 
                     self.wave_grid[pixel_indx[2]:]]
        spectrum = [spectrum[pixel_indx[0]:pixel_indx[1]], 
                    spectrum[pixel_indx[1]:pixel_indx[2]], 
                    spectrum[pixel_indx[2]:]]

        # Remove channels without info
        wave_grid = [wave_grid[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
        starting_indices = [self.starting_indices[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
        pixel_indx = [pixel_indx[i] for i in range(len(spectrum)) if np.std(spectrum[i])>self.std_min]
        spectrum = [spec for spec in spectrum if np.std(spec)>self.std_min]

        # Select chunks from all channels
        spec_chunks = []
        wave_chunks = []
        centre_waves = []
        pixel_indices = []
        for wave_channel, start_is, channel_indx, spec_channel in zip(wave_grid, starting_indices, pixel_indx, spectrum):
            if self.random_chunk:
                for start_indx in start_is:
                    spec_chunks.append(spec_channel[start_indx:start_indx+
                                                    self.num_fluxes])
                    wave_chunks.append(wave_channel[start_indx:start_indx+
                                                    self.num_fluxes])
                    pixel_indices.append(channel_indx + start_indx)
                    centre_waves.append(np.median(wave_chunks[-1]))
            else:
                spec_chunks.append(spec_channel[start_indx:start_indx+self.num_fluxes])
                wave_chunks.append(wave_channel[start_indx:start_indx+self.num_fluxes])
                pixel_indices.append(channel_indx + start_indx)
                centre_waves.append(np.median(wave_chunks[-1]))

        # Perform augmentations according to tasks
        task_labels = []
        for t, tm, ts in zip(self.tasks, self.task_means, self.task_stds):
            if t.lower()=='wavelength':
                task_labels.append(centre_waves) 

        if len(task_labels)>0:
            task_labels = torch.from_numpy(np.vstack(task_labels).T.astype(np.float32))

        spec_chunks = torch.from_numpy(np.vstack(spec_chunks).astype(np.float32))
        wave_chunks = torch.from_numpy(np.vstack(wave_chunks).astype(np.float32))
        pixel_indices = torch.from_numpy(np.vstack(pixel_indices).astype(int))

        return {'spectrum chunks':spec_chunks,
                'multimodal labels':multimodal_labels,
                'unimodal labels':unimodal_labels,
                'task labels':task_labels,
                'pixel_indx':pixel_indices}