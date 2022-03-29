import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

icd_filter = ['48240', '48241', '48242']

def main():
    print('Reading diagnoses_icd...')
    columns = ['subject_id', 'hadm_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing staphylococcus...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.subject_id == subject] for subject in data.subject_id.unique()]])

    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/staphylococcus.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate staphylococcus!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_patient(chunk):
    chunk['staphylococcus'] = chunk.icd_code.isin(icd_filter).astype(int)
    chunk = chunk.drop(columns=['subject_id', 'icd_code'])
    chunk = chunk.drop_duplicates(['hadm_id', 'staphylococcus'])
    if chunk.values.any():
        id = chunk.staphylococcus.idxmax()
        before = chunk.iloc[:id, :]
        after = chunk.iloc[id:, :]
        after = after.assign(staphylococcus=1)
        chunk = pd.concat([before, after])

    chunk = chunk.set_index('hadm_id')
    chunk = chunk[~chunk.index.duplicated(keep='last')] # Sometimes an admission will have 0 then 1 if there was another diagnosis prior to staphylococcus
    return chunk

if __name__ == '__main__':
    print('Saving staphylococcus feature...')
    main().to_pickle(feature_root + '/staphylococcus.pickle')
    print('Saved staphylococcus!\n')
