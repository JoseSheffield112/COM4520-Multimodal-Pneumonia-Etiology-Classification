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
    wbc = np.empty((0,2), float)
    print('Patient count: ', len(set(data.index.values)))
    data = data.drop(columns=['hour'])
    # making array here
    for subject in set(data.index.values):
        # loading each value in respective variable
        arr = np.empty(2, float)
        chunk = [data[data.index == subject]]
        admission = chunk[0].hadm_id.iloc[0]
        mean = chunk[0].value.iloc[0]
        minimum = chunk[0].Minimum.iloc[0]
        maximum = chunk[0].Maximum.iloc[0]
        # storing them in temp array
        arr = [admission, [minimum,maximum,mean]]
        # adding that to wbc
        wbc = np.append(wbc, np.array([arr]), axis=0)
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
    chunk = chunk.drop(columns=['charttime'])
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    chunk = chunk[(chunk.hour < 24) & (chunk.value.astype(float) > 0) & (chunk.value.astype(float) < 1000)] # in SQL anything above 11 is abnormal - there's actually a patient with 281!
    ##
    values = (chunk.size)/3 # gives me (cols*rows), so divided by 3 since theres still subject id, value, hour
    max = min = chunk.value.iloc[0].astype(float)
    if(values>1): #if more than 1 record
        sum=0.0
        for i in range(0,int(values)): # we iterate over them and get mean
            value = chunk.value.iloc[i].astype(float)
            if(value<min):
                min = value
            elif(value>max):
                max = value
            sum=sum+value
        mean = sum/values
        chunk.value.iloc[0] = mean
    chunk['Minimum'] = min
    chunk['Maximum'] = max
    chunk = chunk.head(1) # dropping other rows
    return chunk

if __name__ == '__main__':
    main()