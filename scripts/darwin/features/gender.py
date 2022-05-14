import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from config.darwin.config import *

def main():
    print('Reading admissions...')
    columns = ['subject_id', 'hadm_id']
    iter_csv = pd.read_csv(origin_root + '/core/admissions.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    admissions = pd.concat([chunk for chunk in iter_csv])

    print('Reading patients...')
    columns = ['subject_id', 'gender']
    iter_csv = pd.read_csv(origin_root + '/core/patients.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    patients = pd.concat([chunk for chunk in iter_csv])
    
    data = pd.merge(admissions, patients, left_index=True, right_index=True)
    data = data.set_index('hadm_id')

    data.gender[data.gender == 'M'] = 0
    data.gender[data.gender == 'F'] = 1
    data.gender = data.gender.astype(float)
    
    if save_intermediates:
        print('Saving intermediate...')
        intermediate_path = Path(intermediate_root + '/gender.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved intermediate gender!')

    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

if __name__ == '__main__':
    print('Saving gender feature...')
    main().to_pickle(feature_root + '/gender.pickle')
