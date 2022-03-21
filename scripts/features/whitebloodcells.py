import numpy as np
import pandas as pd
from pathlib import Path

origin_root = 'datasets/mimic-iv-full-cohort'
interm_root = 'intermediates/'
output_root = 'features/'

itemid_filter = [51300, 51301]

def main():
    print('Reading labevents...')
    columns = ['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/hosp/labevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing whitebloodcells...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])
    
    print('Saving...')
    output_path = Path(interm_root + '/whitebloodcells.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved whitebloodcells!')
    print('Generating npy...')
    whitebloodcells = np.empty((0,24), float)
    print('Patient count: ', len(set(data.index.values)))
    i = 0
    for subject in set(data.index.values):
        if(subject == 17121520):
            continue            
        arr = np.empty(24, float)
        chunk = [data[data.index == subject]]
        for row in chunk:
            arr[row.hour] = row.value
        whitebloodcells = np.append(whitebloodcells, np.array([arr]), axis=0)
    print('Saving whitebloodcells feature...')
    np.save(output_root + '/whitebloodcells.npy', whitebloodcells)
    print('Shape: ', whitebloodcells.shape)
    print('Saved whitebloodcells!\n')
    return whitebloodcells

def process_patient(chunk):
    first = chunk.head(1)
    chunk = chunk[chunk.hadm_id == first.hadm_id.iloc[0]]
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    chunk = chunk.drop(columns=['hadm_id', 'charttime'])
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 0) & (chunk.value.astype(float) < 300)]
    return chunk

if __name__ == '__main__':
    main()


#############################
#bad patients (subject_id):

#### 17121520

# 88356048,17121520,24354186.0,93733660,50882,2186-01-18 10:42:00,2186-01-18 11:39:00,29.0,29.0,mEq/L,22.0,32.0,,STAT,
# 88356056,17121520,24354186.0,93733660,50971,2186-01-18 10:42:00,2186-01-18 11:39:00,3.7,3.7,mEq/L,3.3,5.1,,STAT,
# 88356057,17121520,24354186.0,93733660,50983,2186-01-18 10:42:00,2186-01-18 11:39:00,136.0,136.0,mEq/L,133.0,145.0,,STAT,
# 88356058,17121520,24354186.0,93733660,51006,2186-01-18 10:42:00,2186-01-18 11:39:00,23.0,23.0,mg/dL,6.0,20.0,abnormal,STAT,
# 88356079,17121520,24354186.0,51661119,50882,2186-01-19 00:00:00,2186-01-19 01:03:00,28.0,28.0,mEq/L,22.0,32.0,,ROUTINE,
# 88356086,17121520,24354186.0,51661119,50971,2186-01-19 00:00:00,2186-01-19 01:03:00,3.3,3.3,mEq/L,3.3,5.1,,ROUTINE,
# 88356087,17121520,24354186.0,51661119,50983,2186-01-19 00:00:00,2186-01-19 01:03:00,132.0,132.0,mEq/L,133.0,145.0,abnormal,ROUTINE,
# 88356088,17121520,24354186.0,51661119,51006,2186-01-19 00:00:00,2186-01-19 01:03:00,23.0,23.0,mg/dL,6.0,20.0,abnormal,ROUTINE,
# 88356116,17121520,24354186.0,68303455,50882,2186-01-19 22:27:00,2186-01-20 01:32:00,28.0,28.0,mEq/L,22.0,32.0,,ROUTINE,
# 88356123,17121520,24354186.0,68303455,50971,2186-01-19 22:27:00,2186-01-20 01:32:00,3.5,3.5,mEq/L,3.3,5.1,,ROUTINE,
# 88356124,17121520,24354186.0,68303455,50983,2186-01-19 22:27:00,2186-01-20 01:32:00,134.0,134.0,mEq/L,133.0,145.0,,ROUTINE,
# 88356125,17121520,24354186.0,68303455,51006,2186-01-19 22:27:00,2186-01-20 01:32:00,22.0,22.0,mg/dL,6.0,20.0,abnormal,ROUTINE,

# 88356114,17121520,24354186.0,55626272,51301,2186-01-19 22:27:00,2186-01-20 02:33:00,4.8,4.8,K/uL,4.0,11.0,,ROUTINE,CHECKED FOR NRBCS. #1

# 88356044,17121520,24354186.0,2661227,51301,2186-01-18 10:42:00,2186-01-18 12:09:00,8.8,8.8,K/uL,4.0,11.0,,STAT,VERIFIED BY SMEAR.  CHECKED FOR NRBCS. #2

# 88356077,17121520,24354186.0,44421159,51301,2186-01-19 00:00:00,2186-01-19 01:22:00,5.6,5.6,K/uL,4.0,11.0,,ROUTINE,VERIFIED.  CHECKED FOR NRBC. #3

## Current issue:
## When trying to handle this patient, it gets a value for time of -36;
##-36 is acquired because we go from #1 to #2, so the tables are not arranged chronologically

## Plus, it seems interpolation might be necessary as there's very few values
###### HOWEVER - I've looked at this patient in postgres and there's 290 rows of values for (51300, 51301) - Did I run the scripts wrong, or might there be an issue with the csv file pruning?
