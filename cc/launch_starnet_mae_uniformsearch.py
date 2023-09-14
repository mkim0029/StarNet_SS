import os
import itertools
import numpy as np

# Starting number for jobs
start_model_num = 327
num_models = 50

# [Min, max, num_decimals]
uniform_params = {'bs':[128, 512,0],
                  'lr': [0.0001, 0.01, 4],
                  'lrf': [100, 10000, -2],
                  'mr': [0.2, 0.8, 2],
                  'tlw': [0.5,10,1],
                  'wd': [0.001, 0.02, 3],
                  'lplr': [0.0001, 0.01, 4],
                  'lplrf': [100, 10000, -2],
                  'lpbs': [128, 512, 0],
                  'lpdo': [0, 0.15, 2],
                  'mnf': [0.01,0.15,2]}

static_params = {'n': 1}

# Create a list of all possible parameter combinations
#keys, values = zip(*grid_params.items())
#param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
print('Launching %i models' % (num_models))

# Each model will draw all of the parameters from a uniform distribution
for model_num in range(start_model_num, start_model_num+num_models):
    param_cmd = ''
    for k, vals in uniform_params.items():
        # Sample from uniform distbn
        val = np.random.uniform(vals[0], vals[1])
        # Round to correct decimals
        val = np.round(val, vals[2])
        if vals[2]<=0:
            val=int(val)
        param_cmd += '-%s %s '% (k, val)
    for k, val in static_params.items():
        param_cmd += '-%s %s '% (k, val)
    
    launch_cmd = ('python launch_starnet_mae.py starnet_mae_%i' % (model_num) +
                  ' %s -co " %s"' % (param_cmd, param_cmd))
    print(launch_cmd)
    
    
    # Execute jobs
    os.system(launch_cmd)
