import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

itemid_filter = [211, 220045]

def main():
    print('Reading chartevents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing heartrates...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_admission(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])
    
    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/heartrates.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate heartrates!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_admission(chunk):
    first = chunk.head(1)
    first = first.drop(columns='value')

    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    chunk = chunk.drop(columns='charttime')
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 0) & (chunk.value.astype(float) < 300)]
    
    heartrates = np.empty(24, float)
    for _, row in chunk.iterrows():
        heartrates[int(row.hour)] = row.value
    first = first.drop(columns='charttime')
    first['heartrates'] = [heartrates]
    return first

if __name__ == '__main__':
    print('Saving heartrates feature...')
    main().to_pickle(feature_root + '/heartrates.pickle')
    print('Saved heartrates!\n')