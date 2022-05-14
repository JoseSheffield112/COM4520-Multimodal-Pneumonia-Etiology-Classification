import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from darwin.config import *

MIN_READINGS_TO_KEEP = 5

itemid_filter = [676, 678, 223761, 223762]

def main():
    print('Reading chartevents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing temperatures...')
    data = pd.concat([process_admission(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])
    
    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/temperatures.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate temperatures!')

    print('Patient count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_admission(chunk):
    first = chunk.head(1)
    first = first.drop(columns=['itemid', 'value'])

    chunk['hour'] = ((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600
    chunk = chunk.drop(columns='charttime')
    first = first.drop(columns='charttime')
    chunk = chunk[~chunk.hour.duplicated(keep='first')]

    centigrades = chunk[chunk.itemid.isin([676, 223762])]
    fahrenheits = chunk[chunk.itemid.isin([678, 223761])]
    fahrenheits.value = ((fahrenheits.value.astype(float) - 32) * 5 / 9)
    chunk = pd.concat([centigrades, fahrenheits])
    chunk = chunk.drop(columns='itemid')
    
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 25) & (chunk.value.astype(float) < 50)]
    
    x = chunk.hour.values.astype(float)
    y = chunk.value.values.astype(float)

    if len(x) >= MIN_READINGS_TO_KEEP:
        first['temperatures'] = [np.interp(range(24), x, y).round(1)]
    else:
        first['temperatures'] = [np.nan]
    return first

if __name__ == '__main__':
    print('Saving temperatures feature...')
    main().to_pickle(feature_root + '/temperatures.pickle')
    print('Saved temperatures!\n')
