import numpy as np
import pandas as pd
from pathlib import Path
from config import *

icd_filter = [i for i in range(1960, 1992)]
icd_filter += [i for i in range(20970, 20976)]
icd_filter += [20979, 78951]

def main():
    print('Reading diagnoses_icd...')
    columns = ['subject_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing mscancerresults...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])

    print('Saving...')
    intermediate_path = Path(intermediate_root + '/mscancerresults.csv')  
    intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(intermediate_path)
    print('Saved mscancerresults!')
    print('Generating npy...')
    mscancer = np.empty((0), float)
    print('Patient count: ', len(set(data.index.values)))
    mscancer = data.values.astype(int)
    print('Saving metastatic cancer feature...')
    np.save(feature_root + '/mscancer.npy', mscancer)
    print('Shape: ', mscancer.shape)
    print('Saved mscancer!\n')
    return mscancer

def process_patient(chunk):
    chunk['has_mscancer'] = chunk.isin(icd_filter)
    chunk = chunk.drop(columns='icd_code')
    chunk = chunk[~chunk.has_mscancer.duplicated(keep='first')]
    if True in chunk.values:
        chunk = chunk[chunk.has_mscancer == True]
    return chunk

if __name__ == '__main__':
    main()
