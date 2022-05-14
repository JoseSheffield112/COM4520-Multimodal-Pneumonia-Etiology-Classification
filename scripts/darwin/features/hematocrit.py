import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from config.darwin.config import *

itemid_filter = [50810, 51221]

def main():
    print('Reading labeveents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/hosp/labevents.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])

    print('Processing hematocrit...')
    data = pd.concat([process_admissions(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()] if type(chunk) is not type(None)])

    ## Intermediate results
    if save_intermediates:
        intermediate_path = Path(intermediate_root + '/hematocrit.csv')  
        intermediate_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(intermediate_path)
        print('Saved hematocrit!')

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
    if values > 0:
        maximum = minimum = mean = chunk.value.iloc[0]#.astype(float)
        if(values>1): #if more than 1 record
            sum=0.0
            for i in range(0,int(values)): # we iterate over them and get mean
                value = chunk.value.iloc[i]#.astype(float)
                if(float(value)<float(minimum)):
                    minimum = value
                elif(float(value)>float(maximum)):
                    maximum = value
                sum=sum+float(value)
            mean = sum/values
            chunk.value.iloc[0] = mean

        ## Final array
        temp = np.array([minimum, maximum, mean]).astype(float)
        if np.isnan(temp).any():
            first['hematocrit'] = np.nan
        else:
            first['hematocrit'] = [temp]
        return first

if __name__ == '__main__':
    print('Saving hematocrit feature...')
    main().to_pickle(feature_root + '/hematocrit.pickle')
    print('Saved hematocrit!\n')