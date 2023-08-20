import os
from string import ascii_lowercase

# Number for model name
model_num = 235

# Number of models to train with identical parameters
num_ensemble = 6

# Parameters to change from default configuration
params = {'acc':'def-bazalova'}

# Turn parameters into command
param_cmd = ''
for k in params:
    if (hasattr(params[k], '__len__')) and (not isinstance(params[k], str)):
        if len(params[k])==0:
            param_cmd += '-%s [] '% (k)
        elif isinstance(params[k][0], str):
            param_cmd += '-%s %s '% (k, ' '.join(str(val) for val in params[k]))
        else:
            param_cmd += '-%s [%s] '% (k, ','.join(str(val) for val in params[k]))
    else:
        param_cmd += '-%s %s '% (k, params[k])

# Launch ensemble
for i, e in zip(range(num_ensemble), ascii_lowercase):
    model_name = 'starnet_mae_%i_%s' % (model_num, e)
    
    launch_cmd = ('python launch_starnet_mae.py %s %s -co "Ensemble with %s"' % (model_name, 
                                                                                 param_cmd, 
                                                                                 param_cmd))
    print(launch_cmd)
        
    # Execute jobs
    os.system(launch_cmd)
