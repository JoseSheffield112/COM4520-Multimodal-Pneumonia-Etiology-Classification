from os import listdir
from importlib import import_module
import numpy as np
import config

save_npz = True
low_memory = False
in_data_root = 'datasets/mimic-iv/mimic-iv-1.0'
out_data_root = 'datasets/mimic-iv/mimic-iv-new'
origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort'
feature_script_root = 'scripts/features'
intermediate_root = 'intermediates/'
feature_root = 'features'
output_root = 'output'


features = {}

for filename in listdir(feature_script_root):
    if filename.endswith('.py'):
        file = filename[:-3] # No extension
        print('Running ', file)
        feature = import_module('.' + file, 'features').main()
        if save_npz and not low_memory:
            features[file] = feature
        else:
            features[file] = None

if save_npz:
    print('Saving compiled output...')
    if low_memory:
        for filename in listdir(feature_root):
            if filename.endswith('.npy'):
                file = filename[:-4] # No extension
                features[file] = np.load(feature_root + '/' + filename)
                
    else:
        np.savez(output_root + '/features.npz', **features)
    print('Saved compiled output!\n')

print('Features:', ', '.join(features.keys()))