import numpy as np
import pandas as pd
from pathlib import Path

origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort'
interm_root = 'intermediates/'
output_root = 'features/'

def main():
    print('Reading diagnoses_icd...')
    columns = ['subject_id', 'icd_code']
    iter_csv = pd.read_csv(origin_root + '/hosp/diagnoses_icd.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk for chunk in iter_csv])

    print('Processing aidsresults...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])

    print('Saving...')
    output_path = Path(interm_root + '/aidsresults.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved aidstesults!')
    print('Generating npy...')
    aids = np.empty((0), float)
    print('Patient count: ', len(set(data.index.values)))
    aids = data.values.astype(int)
    print('Saving aids feature...')
    np.save(output_root + '/aids.npy', aids)
    print('Shape: ', aids.shape)
    print('Saved aids!\n')
    return aids

def process_patient(chunk):
    icd_filter = ['042', '043', '044', '0449']
    chunk['has_aids'] = chunk.isin(icd_filter)
    chunk = chunk.drop(columns='icd_code')
    chunk = chunk[~chunk.has_aids.duplicated(keep='first')]
    if True in chunk.values:
        chunk = chunk[chunk.has_aids == True]
    return chunk

if __name__ == '__main__':
    main()
