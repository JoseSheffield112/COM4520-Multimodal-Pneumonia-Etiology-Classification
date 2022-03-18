from os import listdir
from importlib import import_module
import numpy as np

save_npz = True
feature_root = 'scripts/features'
output_root = 'output'


features = {}

for filename in listdir(feature_root):
    if filename.endswith('.py'):
        file = filename[:-3] # No extension
        print('Running ', file)
        feature = import_module('.' + file, 'features').main()
        if save_npz:
            features[file] = feature
        else:
            features[file] = None

if save_npz:
    print('Saving compiled output...')
    np.savez(output_root + '/features.npz', **features)
    print('Saved compiled output!\n')

print('Features:', ', '.join(features.keys()))