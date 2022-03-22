from os import listdir
from importlib import import_module
import numpy as np
from config import *

features = {}
failed = {}

for filename in listdir(feature_script_root):
    if filename.endswith('.py'):
        file = filename[:-3] # No extension
        print('Running ', file)
        try:
            if save_npz and not low_memory:
                features[file] = import_module('.' + file, 'features').main()
            else:
                import_module('.' + file, 'features').main()
                features[file] = None
        except Exception as e:
            failed[file] = e

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
if failed:
    print('The following features failed:')
    for key in failed:
        print(key, failed[key])