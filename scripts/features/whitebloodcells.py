## @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ | @NOTICE@ ##
## This script cannot handled missing hadm_id entries - which mimic-iv-full-cohort-v2 has
import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

itemid_filter = [51300, 51301]

def main():
    print('Reading labeveents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/hosp/labevents.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing whitebloodcells...')
    data = pd.concat([process_admissions(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])

    ## Intermediate results
    if save_intermediates:
        intermediate_path = Path(intermediate_root + '/whitebloodcells.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved whitebloodcells!')

    print('Patient count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    
    return data

def process_admissions(chunk):
    chunk.sort_values(by=['charttime'], ascending = True, inplace=True) # pandas doesn't like this, but i don't like the fact it doesn't like it without offering me an alternative so left it here
    first = chunk.head(1)
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)

    first = first.drop(columns=['itemid', 'value', 'charttime'])
    chunk = chunk.drop(columns=['itemid', 'charttime'])
    
    chunk = chunk[~chunk.hour.duplicated(keep='first')]
    
    ## Calculating min/max/mean
    values = (chunk.size)/3 # gives me (cols*rows), so divided by 3 since theres still subject id, value, hour
    maximum = minimum = mean = chunk.value.iloc[0].astype(float)
    if(values>1): #if more than 1 record
        sum=0.0
        for i in range(0,int(values)): # we iterate over them and get mean
            value = chunk.value.iloc[i].astype(float)
            if(value<minimum):
                minimum = value
            elif(value>maximum):
                maximum = value
            sum=sum+value
        mean = sum/values
        chunk.value.iloc[0] = mean

    ## Final array
    temp = np.array([minimum, maximum, mean])
    first['wbcount'] = [temp]
    return first

if __name__ == '__main__':
    print('Saving temperatures feature...')
    main().to_pickle(feature_root + '/temperatures.pickle')
    print('Saved temperatures!\n')