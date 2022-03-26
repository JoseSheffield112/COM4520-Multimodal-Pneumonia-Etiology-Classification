import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

def main():
    print('Reading admissions...')
    columns = ['subject_id', 'hadm_id']
    iter_csv = pd.read_csv(origin_root + '/core/admissions.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    admissions = pd.concat([chunk for chunk in iter_csv])

    print('Reading patients...')
    columns = ['subject_id', 'anchor_age']
    iter_csv = pd.read_csv(origin_root + '/core/patients.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    patients = pd.concat([chunk for chunk in iter_csv])
    
    data = pd.merge(admissions, patients, left_index=True, right_index=True)
    data = data.set_index('hadm_id')
    data.columns = ['age']
    
    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/age.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate age!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    print(data)
    return data

if __name__ == '__main__':
    print('Saving age feature...')
    main().to_pickle(feature_root + '/age.pickle')
    print('Saved age!\n')
    f = pd.read_pickle(feature_root + '/age.pickle')
    print(f)