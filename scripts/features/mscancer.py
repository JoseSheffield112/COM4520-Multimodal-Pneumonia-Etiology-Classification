import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

icd_filter = [str(i) for i in range(1960, 1992)]
icd_filter += [str(i) for i in range(20970, 20976)]
icd_filter += ['20979', '78951']

def main():
    print('Reading diagnoses_icd...')
    columns = ['subject_id', 'hadm_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing mscancer...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.subject_id == subject] for subject in data.subject_id.unique()]])

    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/mscancer.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate mscancer!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    print('Saved mscancer!\n')
    return data

def process_patient(chunk):
    chunk['has_mscancer'] = chunk.icd_code.isin(icd_filter).astype(int)
    chunk = chunk.drop(columns=['subject_id', 'icd_code'])
    chunk = chunk.drop_duplicates(['hadm_id', 'has_mscancer'])
    if chunk.values.any():
        id = chunk.has_mscancer.idxmax()
        before = chunk.iloc[:id-1, :]
        after = chunk.iloc[id-1:, :]
        after = after.assign(has_mscancer=1)
        chunk = pd.concat([before, after])

    chunk = chunk.set_index('hadm_id')
    chunk = chunk[~chunk.index.duplicated(keep='last')] # Sometimes an admission will have 0 then 1 if there was another diagnosis prior to metastatic cancer
    return chunk

if __name__ == '__main__':
    main()
