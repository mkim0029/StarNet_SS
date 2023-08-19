import os
import itertools

# Starting number for jobs
start_model_num = 162
'''
# Different parameters to try out 
# (check the launch script to see what these keys correspond to)
grid_params = {'jt': [2],
               'upa': ['True'],
               'pan':['starnet_mae_111'],
               'lpop':['adamw'],
               'lplr': [0.1, 0.3], # 0.01
               'lplrf': [100., 500], # 10, 1000
               'lpdo': [0.95], # 0.8, 0.9
              'lpbs': [1024, 2048]} # 256
'''

grid_params = {'bs': [64, 128],
              'lr': [0.01, 0.001],
              'lrf': [1000, 100],
              'wd': [0.05, 0.01],
              'mr': [0.75, 0.5],}

# Create a list of all possible parameter combinations
keys, values = zip(*grid_params.items())
param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
print('Launching %i models' % (len(param_combos)))

# Launch a model for each combination
model_num = start_model_num
for params in param_combos:
    param_cmd = ''
    for k in params:
        if (hasattr(params[k], '__len__')) and (not isinstance(params[k], str)):
            for k_ in k.split('/'):
                if len(params[k])==0:
                    param_cmd += '-%s [] '% (k)
                elif isinstance(params[k][0], str):
                    param_cmd += '-%s %s '% (k_, ' '.join(str(val) for val in  params[k]))
                else:
                    param_cmd += '-%s [%s] '% (k_, ','.join(str(val) for val in  params[k]))
        else:
            for k_ in k.split('/'):
                param_cmd += '-%s %s '% (k_, params[k])
    
    launch_cmd = ('python launch_starnet_mae.py starnet_mae_%i' % (model_num) +
                  ' %s -co " %s"' % (param_cmd, param_cmd))
    print(launch_cmd)
    
    model_num += 1
    
    # Execute jobs
    os.system(launch_cmd)
