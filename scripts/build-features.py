from copyreg import pickle
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
        try:
            table[column] = import_module('.' + column, 'preprocessing').main(table[column].values)
            print('success!')
        except Exception as e:
            print('failed:', e)
    print()
    return table

def format_timeseries(table, columns):
    table = table.drop(columns=[col for col in list(table) if col not in columns])
    arr = table.to_numpy()
    arr = np.array([row.tolist() for row in arr.flatten()]).reshape(arr.shape[0], -1, arr.shape[1])
    return arr

def format_static(table, columns):
    table = table.drop(columns=[col for col in list(table) if col not in columns])
    arr = table.to_numpy()
    arr = np.array([np.hstack(row) for row in arr])
    return arr

if __name__=='__main__':
    features = [feature for feature in get_individual_features() if type(feature) is type(pd.DataFrame())]

    features = pd.concat(features, axis=1)
    features = features.dropna(thresh=4) # Keep records with {thresh} non-NaN columns, not including hadm_id

    features = preprocess(features)

    print('Saving data csv...')
    csv_path = Path(output_root + '/data.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(csv_path)
    print('Saved data csv!')

    print('Creating test...')
    test_set = pd.read_csv(labels_root + '/test.csv', header=0, index_col=[0], usecols=['hadm_id', 'etiology'])
    test_table = pd.merge(features, test_set, left_index=True, right_index=True)
    test_ts = format_timeseries(test_table, ['heartrates', 'systolic_blood_pressure', 'temperatures'])
    print('Timeseries shape:', test_ts.shape)
    test_static = format_static(test_table, ['aids', 'mscancer', 'whitebloodcells'])
    print('Static shape:', test_static.shape)
    etiologies = test_table.etiology.values
    test_labels = etiologies.reshape(etiologies.shape[0], 1)
    print(type(test_labels), test_labels.shape)
    print('Labels shape:', test_labels.shape)
    print('Saving test csv...')
    csv_path = Path(output_root + '/test.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(csv_path)
    print('Saved test csv!')

    print('Creating train...')
    train_set = pd.read_csv(labels_root + '/train.csv', header=0, index_col=[0], usecols=['hadm_id', 'etiology'])
    train_table = pd.merge(features, train_set, left_index=True, right_index=True)
    train_ts = format_timeseries(train_table, ['heartrates', 'systolic_blood_pressure', 'temperatures'])
    print('Timeseries shape:', train_ts.shape)
    train_static = format_static(train_table, ['aids', 'mscancer', 'whitebloodcells'])
    print('Static shape:', train_static.shape)
    etiologies = train_table.etiology.values
    train_labels = etiologies.reshape(etiologies.shape[0], 1)
    print('Labels shape:', train_labels.shape)
    print('Saving train csv...')
    csv_path = Path(output_root + '/train.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_csv(csv_path)
    print('Saved train csv!')

    if save_npz:
        print('Saving output im.npz...')
        test_array = {
            'timeseries': test_ts,
            'static': test_static,
            'labels': test_labels
        }
        print('Test shapes:', *[test_array[arr].shape for arr in test_array])
        train_array = {
            'timeseries': train_ts,
            'static': train_static,
            'labels': train_labels
        }
        print('Train shapes:', *[train_array[arr].shape for arr in train_array])

        impk_path = Path(output_root + '/im.npz')  
        impk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(impk_path, test=test_array, train=train_array)
        print('Saved output im.npz!')

        impk = np.load(impk_path, allow_pickle=True)
        print('\nFirst test', impk['test'].timeseries, sep='\n')
        print('\nFirst train', [arr[:5] for arr in impk['train']], sep='\n')
