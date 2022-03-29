#from copyreg import pickle
import pickle
import os
import sys
from tkinter import image_names
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
        #print(table[column].values)
        table[column] = import_module('.' + column, 'preprocessing').main(table[column].values)
        print('success!')
        #except Exception as e:
        #   print('failed:', e)
    print()
    return table

def format_timeseries(table):
    columns = ['heartrates', 'systolic_blood_pressure', 'temperatures']
    table = table.drop(columns=[col for col in list(table) if col not in columns])
    arr = table.to_numpy()
    arr = np.array([row.tolist() for row in arr.flatten()]).reshape(arr.shape[0], -1, arr.shape[1])
    return arr

def format_static(table):
    columns = ['age', 'aids', 'gender', 'mscancer', 'mycoplasma', 'staphylococcus' 'whitebloodcells']
    table = table.drop(columns=[col for col in list(table) if col not in columns])
    arr = table.to_numpy()
    arr = np.array([np.hstack(row) for row in arr])
    return arr

def get_image_data():
    #Gets image data formatted as a dataframe with the image input as one column and hadm_id as the other
    
    f = open(image_data_root + '/train.pk', 'rb')
    train_images = pickle.load(f)
    f.close()
    train_images = [(sample['hadm_id'].item(),sample['img'].detach().cpu().numpy()) for sample in train_images]
    train_images = pd.DataFrame(train_images,columns = ['hadm_id','image']).set_index('hadm_id')

    f = open(image_data_root + '/valid.pk', 'rb')
    valid_images = pickle.load(f)
    f.close()
    valid_images = [(sample['hadm_id'].item(),sample['img'].detach().cpu().numpy()) for sample in valid_images]
    valid_images = pd.DataFrame(valid_images,columns = ['hadm_id','image']).set_index('hadm_id')

    f = open(image_data_root + '/test.pk', 'rb')
    test_images = pickle.load(f)
    f.close()
    test_images = [(sample['hadm_id'].item(),sample['img'].detach().cpu().numpy()) for sample in test_images]
    test_images = pd.DataFrame(test_images, columns = ['hadm_id','image']).set_index('hadm_id')

    return {'train':train_images,'valid':valid_images,'test':test_images}




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

    images = get_image_data()

    print('Creating test...')
    test_set = pd.read_csv(labels_root + '/test.csv', header=0, index_col=[0], usecols=['hadm_id', 'etiology']).astype({'etiology': 'int32'})
    test_table = pd.merge(features, test_set, left_index=True, right_index=True)
    test_table = pd.merge(test_table, images['test'], left_index=True, right_index=True)

    test_ts = format_timeseries(test_table)
    print('Timeseries shape:', test_ts.shape)
    test_static = format_static(test_table)
    print('Static shape:', test_static.shape)
    test_images = test_table.image.values
    print('Images shape:', test_images.shape)
    test_labels = test_table.etiology.values
    print('Labels shape:', test_labels.shape)
    print('Saving test csv...')
    csv_path = Path(output_root + '/test.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(csv_path)
    print('Saved test csv!')

    print('Creating train...')
    train_set = pd.read_csv(labels_root + '/train.csv', header=0, index_col=[0], usecols=['hadm_id', 'etiology']).astype({'etiology': 'int32'})
    train_table = pd.merge(features, train_set, left_index=True, right_index=True)
    train_table = pd.merge(train_table, images['train'], left_index=True, right_index=True)

    train_ts = format_timeseries(train_table)
    print('Timeseries shape:', train_ts.shape)
    train_static = format_static(train_table)
    print('Static shape:', train_static.shape)
    train_images = train_table.image.values
    print('Images shape:', train_images.shape)
    train_labels = train_table.etiology.values
    print('Labels shape:', train_labels.shape)
    print('Saving train csv...')
    csv_path = Path(output_root + '/train.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_csv(csv_path)
    print('Saved train csv!')

    print('Creating valid...')
    valid_set = pd.read_csv(labels_root + '/valid.csv', header=0, index_col=[0], usecols=['hadm_id', 'etiology']).astype({'etiology': 'int32'})
    valid_table = pd.merge(features, valid_set, left_index=True, right_index=True)
    valid_table = pd.merge(valid_table, images['valid'], left_index=True, right_index=True)

    valid_ts = format_timeseries(valid_table)
    print('Timeseries shape:', valid_ts.shape)
    valid_static = format_static(valid_table)
    print('Static shape:', valid_static.shape)
    valid_images = valid_table.image.values
    print('Images shape:', valid_images.shape)
    valid_labels = valid_table.etiology.values
    print('Labels shape:', valid_labels.shape)
    print('Saving valid csv...')
    csv_path = Path(output_root + '/valid.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_table.to_csv(csv_path)
    print('Saved valid csv!')

    if save_npz:
        print('Saving output im.npz...')
        test_array = {
            'timeseries': test_ts,
            'static': test_static,
            'image' : test_images,
            'labels': test_labels
        }
        print('Test shapes:', *[test_array[arr].shape for arr in test_array])
        train_array = {
            'timeseries': train_ts,
            'static': train_static,
            'image' : train_images,
            'labels': train_labels
        }
        print('Train shapes:', *[train_array[arr].shape for arr in train_array])
        valid_array = {
            'timeseries': valid_ts,
            'static': valid_static,
            'image' : valid_images,
            'labels': valid_labels
        }
        print('Valid shapes:', *[valid_array[arr].shape for arr in valid_array])

        impk_path = Path(output_root + '/im.npz')  
        impk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(impk_path, test=test_array, train=train_array)
        pickle.dump( {'test':test_array, 'train':train_array, 'valid':valid_array}, open(output_root + '/im.pk', 'wb'))
        print('Saved output im.npz!')

        impk = pickle.load(open(output_root + '/im.pk', 'rb'))
        print('\nFirst test', *[impk['test'][key][0] for key in impk['test']], sep='\n')
        # print('\nFirst train', *[impk['train'][key][0] for key in impk['train']], sep='\n')
        # print('\nFirst valid', *[impk['valid'][key][0] for key in impk['valid']], sep='\n')
