import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from darwin.config import *

MIN_READINGS_TO_KEEP = 5

# chartevents table
# 220179
# 220050

itemid_filter = [220179, 220050]

def main():
    print('Reading chartevents...')
    columns = ['hadm_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])
    #print(data)

    print('Processing systolic_blood_pressure...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_admission(chunk) for chunk in [data[data.index == admission] for admission in data.index.unique()]])

    if save_intermediates:
        print('Saving intermediates...')
        output_path = Path(intermediate_root + '/systolic_blood_pressure.csv')  
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(output_path)
        print('Saved intermediate systolic_blood_pressure!')

    #print(data)
    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_admission(chunk):
    # Last thing to do is interpolate
    # Check that average is correct
    #print(chunk)
    first = chunk.head(1)
    first = first.drop(columns='value')

    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    
    chunk = chunk.drop(columns='charttime')
    chunk = chunk[chunk.hour < 24]
    
    chunk.value = pd.to_numeric(chunk.value)
    chunk = chunk[chunk.value < 400.0]
    chunk = chunk[chunk.value > 0.0]

    chunk = chunk.groupby('hour', as_index=False)['value'].mean()
    chunk = chunk.set_index('hour')

    if len(chunk) >= MIN_READINGS_TO_KEEP:
        chunk_interpolated = pd.DataFrame({ 'hour' : range(0, 24, 1)})
        chunk_interpolated = chunk_interpolated.set_index('hour')
        
        chunk_interpolated = chunk_interpolated.reindex()
        chunk_interpolated.loc[chunk.index,'value'] = chunk.value

        chunk_interpolated = chunk_interpolated.interpolate().reset_index()
        chunk_interpolated['value'] = chunk_interpolated['value'].round(1)

        formatted_chunk = pd.DataFrame({'hadm_id' : [first.index.values[0]], 'systolic_blood_pressure' : [chunk_interpolated.value.values]})
        formatted_chunk = formatted_chunk.set_index('hadm_id')

        return formatted_chunk

    formatted_chunk = pd.DataFrame({'hadm_id' : [first.index.values[0]], 'systolic_blood_pressure' : [np.nan]})
    formatted_chunk = formatted_chunk.set_index('hadm_id')

    return formatted_chunk

if __name__ == "__main__":
    print('Saving systolic_blood_pressure feature...')
    main().to_pickle(feature_root + '/systolic_blood_pressure.pickle')
    print('Saved systolic_blood_pressures!\n')
