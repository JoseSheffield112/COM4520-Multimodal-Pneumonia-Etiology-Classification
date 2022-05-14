import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from config.darwin.config import *

MIN_READINGS_TO_KEEP = 5

itemid_filter = [227456]

def main():
    print('Reading chartevents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing albumin...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_admission(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])
    
    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/albumin.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate albumin!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_admission(chunk):
    first = chunk.head(1)
    first = first.drop(columns='value')

    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600)
    chunk = chunk.drop(columns='charttime')
    first = first.drop(columns='charttime')
    chunk = chunk[~chunk.hour.duplicated(keep='first')]

    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 0) & (chunk.value.astype(float) < 300)]
    
    x = chunk.hour.values.astype(float)
    y = chunk.value.values.astype(float)

    if len(x) >= MIN_READINGS_TO_KEEP:
        first['albumin'] = [np.interp(range(24), x, y).round(1)]
    else:
        first['albumin'] = [np.nan]
    return first

if __name__ == '__main__':
    print('Saving albumin feature...')
    main().to_pickle(feature_root + '/albumin.pickle')
    print('Saved albumin!\n')
