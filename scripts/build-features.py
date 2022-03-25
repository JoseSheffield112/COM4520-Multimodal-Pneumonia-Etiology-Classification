import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

from os import listdir
from importlib import import_module
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

def get_individual_features():
    features = {}
    failed = {}

    for filename in listdir(feature_script_root):
        if filename.endswith('.py'):
            file = filename[:-3] # No extension
            if overwrite_cache:
                print('Running ', file)
                try:
                    features[file] = import_module('.' + file, 'features').main()
                    print('Saving', file, 'feature...')
                    features[file].to_pickle(feature_root + '/' + file + '.pickle')
                    print('Saved', file + '!\n')
                except Exception as e:
                    failed[file] = e
            else:
                try:
                    features[file] = pd.read_pickle(feature_root + '/' + file + '.pickle')
                except Exception as e:
                    failed[file] = e
    
    print('Features:', ', '.join(features.keys()))
    if failed:
        print('The following features failed:')
        for key in failed:
            print(key, failed[key])
    print()
    return list(features.values())

def preprocess(table):
    for column in table.columns:
        print('Preprocessing', column, end='... ')
        #try:
       # print(table[column].values)
        table[column] = import_module('.' + column, 'preprocessing').main(table[column].values)
        print('success!')
        #except Exception as e:
         #   print('failed:', e)
    print()
    return table
            

if __name__=='__main__':
    features = [feature for feature in get_individual_features() if type(feature) is type(pd.DataFrame())]

    

    features = pd.concat(features, axis=1)

    #features = features.dropna(thresh=2) # Keep records with {thresh} non-NaN columns, not including hadm_id


    features = preprocess(features)

    print('Saving output csv...')
    csv_path = Path(output_root + '/features.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(csv_path)
    print('Saved output csv!')

    if save_npz:
        print('Saving output im.pk...')
        impk = features.to_numpy()
        impk_path = Path(output_root + '/im.pk')  
        impk_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(csv_path)
        np.save(output_root + '/im.pk', impk)
        print('Saved output im.pk!')
