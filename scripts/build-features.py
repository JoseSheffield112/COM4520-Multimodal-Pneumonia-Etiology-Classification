from os import listdir
from importlib import import_module
import numpy as np

save_npz = True
low_memory = False
feature_script_root = 'scripts/features'
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