import numpy as np
import pandas as pd
from pathlib import Path

origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort'
interm_root = 'intermediates/'
output_root = 'features/'

itemid_filter = [676, 678, 223761, 223762]

def main():
    print('Reading chartevents...')
    columns = ['subject_id', 'stay_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing temperatureevents...')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])

    print('Converting temperatures...')
    centigrades = data[data.itemid.isin([676, 223762])]
    fahrenheits = data[data.itemid.isin([678, 223761])]
    fahrenheits.value = ((fahrenheits.value.astype(float) - 32) * 5 / 9).round(1)
    data = pd.concat([centigrades, fahrenheits])
    data = data.drop(columns='itemid')
    
    print('Saving...')
    output_path = Path(interm_root + '/temperatureevents.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved temperatureevents!')
    print('Generating npy...')
    temperatures = np.empty((0,24), float)
    print('Patient count: ', len(set(data.index.values)))
    for subject in set(data.index.values):
        arr = np.empty(24, float)
        chunk = [data[data.index == subject]]
        for row in chunk:
            arr[row.hour] = row.value
        temperatures = np.append(temperatures, np.array([arr]), axis=0)
    print('Saving temperatures feature...')
    np.save(output_root + '/temperature.npy', temperatures)
    print('Shape: ', temperatures.shape)
    print('Saved temperatures!\n')
    return temperatures

def process_patient(chunk):
    first = chunk.head(1)
    chunk = chunk[chunk.stay_id == first.stay_id.iloc[0]]
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    chunk = chunk.drop(columns=['stay_id', 'charttime'])
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 25) & (chunk.value.astype(float) < 50)]
    return chunk

if __name__ == '__main__':
    main()
