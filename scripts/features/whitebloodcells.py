## TODO list:
# Need to drop everything else but the values
# Need to tie values to a hadm_id

## @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ ##
## This script cannot handled missing hadm_id entries
import numpy as np
import pandas as pd
from pathlib import Path
from config import *

itemid_filter = [51300, 51301]

def main():
    print('Reading labeveents...')
    columns = ['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/hosp/labevents.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing whitebloodcells...')
    data = data.drop(columns=['itemid'])
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])

    print('Saving...')
    intermediate_path = Path(intermediate_root + '/whitebloodcells.csv')  
    intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(intermediate_path)
    print('Saved whitebloodcells!')
    print('Generating npy...')
    wbc = np.empty((0), float)
    print('Patient count: ', len(set(data.index.values)))
    wbc = data.values.astype(int) # need to sort out this base 10 error
    print('Saving white blood cell values...')
    np.save(feature_root + '/wbc.npy', wbc)
    print('Shape: ', wbc.shape)
    print('Saved aids!\n')
    return wbc

def process_patient(chunk):
    chunk.sort_values(by=['charttime'], ascending = True, inplace=True) # pandas doesn't like this, but i don't like the fact it doesn't like it without offering me an alternative so left it here
    first = chunk.head(1)
    chunk = chunk[chunk.hadm_id == first.hadm_id.iloc[0]]
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    chunk = chunk.drop(columns=['charttime']) # TODO - need to remove the subject ID & hour so we only keep WBC for each hadm
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 0) & (chunk.value.astype(float) < 100)] # in SQL anything above 11 is abnormal
    values = (chunk.size)/3#gives me number of cols, so divided by 3 since theres still subject id, value, hour
    if(values>1): #if more than 1 record
        sum=0.0
        for i in range(0,int(values)):
            sum=sum+(chunk.value.iloc[i].astype(float))#need to take value of iloc here not array
        print(sum)
        mean = sum/values
        print(mean)
        chunk.value.iloc[0] = mean
    chunk = chunk.head(1)
    print(chunk)
    return chunk

if __name__ == '__main__':
    main()