## TODO list:
# Need to drop everything else but the values
# Need to tie values to a hadm_id


#print("-x--x--x--x--x--x--x--x--x--x--x-")

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
            arr = np.empty(2, float)
            chunk = [data[data.index == subject]]
            print("chunk:\n")
            print(chunk)
            print(chunk[0].value.iloc[0])
            print(chunk[0].hadm_id.iloc[0])
            # for row in chunk:
            #     print("row: \n")
            #     print(row.value.iloc[0])
            #     print(row.hadm_id.iloc[0])
            wbc = np.append(wbc, np.array([arr]), axis=0)
    # final construction
    #wbc = data.values.astype(int)
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
    values = (chunk.size)/3 # gives me (cols*rows), so divided by 3 since theres still subject id, value, hour
    if(values>1): #if more than 1 record
        sum=0.0
        for i in range(0,int(values)): # we iterate over them and get mean
            sum=sum+(chunk.value.iloc[i].astype(float))
        mean = sum/values
        chunk.value.iloc[0] = mean
    chunk = chunk.head(1) # dropping other rows
    print(chunk)
    return chunk

if __name__ == '__main__':
    main()