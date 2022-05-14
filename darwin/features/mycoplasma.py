import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from darwin.config import *

icd_filter = ['4830']

def main():
    print('Reading diagnoses_icd...')
    columns = ['subject_id', 'hadm_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing mycoplasma...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.subject_id == subject] for subject in data.subject_id.unique()]])

    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/mycoplasma.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate mycoplasma!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_patient(chunk):
    chunk['mycoplasma'] = chunk.icd_code.isin(icd_filter).astype(int)
    chunk = chunk.drop(columns=['subject_id', 'icd_code'])
    chunk = chunk.drop_duplicates(['hadm_id', 'mycoplasma'])
    if chunk.values.any():
        id = chunk.mycoplasma.idxmax()
        before = chunk.iloc[:id, :]
        after = chunk.iloc[id:, :]
        after = after.assign(mycoplasma=1)
        chunk = pd.concat([before, after])

    chunk = chunk.set_index('hadm_id')
    chunk = chunk[~chunk.index.duplicated(keep='last')] # Sometimes an admission will have 0 then 1 if there was another diagnosis prior to mycoplasma pneumoniae
    return chunk

if __name__ == '__main__':
    print('Saving mycoplasma feature...')
    main().to_pickle(feature_root + '/mycoplasma.pickle')
    print('Saved mycoplasma!\n')
