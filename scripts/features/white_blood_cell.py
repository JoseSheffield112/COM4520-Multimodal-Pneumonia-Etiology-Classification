import numpy as np
import pandas as pd
from pathlib import Path

origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort'
interm_root = 'intermediates/'
output_root = 'features/'

def main():
    print('Reading labevents...')
    columns = ['subject_id', 'stay_id', 'charttime', 'itemid', 'value'] # labevent_id,subject_id,hadm_id,specimen_id,itemid,charttime,storetime,value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag,priority,comments
    #711,10000048,,90050443,51301,2126-11-22 20:45:00,2126-11-22 21:32:00,9.6,9.6,K/uL,4,11,,STAT
    #2106,10000473,,48534344,51301,2138-03-15 22:37:00,2138-03-15 23:24:00,11.3,11.3,K/uL,4,10,abnormal,STAT,
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    itemid_filter = [51300, 51301]
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing whitebloodcounts...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])
    
    print('Saving...')
    output_path = Path(interm_root + '/whitebloodcounts.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved whitebloodcounts!')
    print('Generating npy...')
    wbc = np.empty((0,24), float)
    print('Patient count: ', len(set(data.index.values)))
    for subject in set(data.index.values):
        arr = np.empty(24, float)
        chunk = [data[data.index == subject]]
        for row in chunk:
            arr[row.hour] = row.value
        heartrates = np.append(heartrates, np.array([arr]), axis=0)
    print('Saving heartrates feature...')
    np.save(output_root + '/heartrate.npy', heartrates)
    print('Shape: ', heartrates.shape)
    print('Saved heartrates!\n')
    return heartrates

def process_patient(chunk):
    first = chunk.head(1)
    chunk = chunk[chunk.stay_id == first.stay_id.iloc[0]]
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    chunk = chunk.drop(columns=['stay_id', 'charttime'])
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[chunk.hour < 24]
    return chunk

if __name__ == '__main__':
    main()
