import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

icd_filter = ['4801']

def main():
    print('Reading diagnoses_icd...')
    columns = ['hadm_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing rsv...')
    data = pd.concat([process_admission(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])

    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/rsv.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate rsv!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_admission(chunk):
    chunk['rsv'] = chunk.icd_code.isin(icd_filter).astype(int)
    chunk = chunk.drop(columns='icd_code')
    chunk = chunk.drop_duplicates('rsv')
    if chunk.values.any():
        chunk = chunk.assign(rsv=1)

    chunk = chunk[~chunk.index.duplicated(keep='first')]
    return chunk

if __name__ == '__main__':
    print('Saving rsv feature...')
    main().to_pickle(feature_root + '/rsv.pickle')
    print('Saved rsv!\n')
