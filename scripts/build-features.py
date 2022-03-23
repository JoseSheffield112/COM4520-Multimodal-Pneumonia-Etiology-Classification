import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

from os import listdir
from importlib import import_module
import numpy as np
import pandas as pd
from scripts.config import *



def get_feature_dict():
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
                    print('Saving', file, 'feature...')
                    import_module('.' + file, 'features').main().to_pickle(feature_root + '/' + file + '.pickle')
                    print('Saved', file + '!\n')
                    features[file] = None
            except Exception as e:
                failed[file] = e
    return features, failed

def load_features(features):
    for key in features:
        if not features[key]:
            try:
                features[key] = pd.read_pickle(feature_root + '/' + key + '.pickle')
            except Exception as e:
                print(e)
    return features

# if save_npz:
#     print('Saving compiled output...')
#     if low_memory:
#         for filename in listdir(feature_root):
#             if filename.endswith('.npy'):
#                 file = filename[:-4] # No extension
#                 features[file] = np.load(feature_root + '/' + filename)
#
#     else:
#         np.savez(output_root + '/features.npz', **features)
#     print('Saved compiled output!\n')

if __name__=='__main__':
    features, failed = get_feature_dict()
    print('Features:', ', '.join(features.keys()))

    if failed:
        print('The following features failed:')
        for key in failed:
            print(key, failed[key])

    features = load_features(features)
    
    