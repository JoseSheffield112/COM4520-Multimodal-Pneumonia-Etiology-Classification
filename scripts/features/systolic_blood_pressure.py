import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

# chartevents table
# 220179
# 220050

itemid_filter = [220179, 220050]

def main():
    print('Reading chartevents...')
    columns = ['subject_id', 'stay_id', 'charttime', 'itemid', 'value']
    iter_csv = pd.read_csv(origin_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])
    print(data)

    print('Processing systolicbloodpressureevents...')
    data = data.drop(columns='itemid')
    data = pd.concat([process_patient(chunk) for chunk in [data[data.index == subject] for subject in data.index.unique()]])

    """ if save_intermediates:
        print('Saving intermediates...')
        output_path = Path(interm_root + '/systolicbloodpressureevents.csv')  
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        data.to_csv(output_path)
        print('Saved systolicbloodpressureevents!')
    """
    print(data)
    print('Admission count: ', len(set(data.index.values)))
    print('Shape: ', data.shape)
    return data

def process_patient(chunk):
    # Last thing to do is interpolate
    # Check that average is correct
    #print(chunk)
    first = chunk.head(1)
    first = first.drop(columns='value')

    chunk = chunk[chunk.stay_id == first.stay_id.iloc[0]]
    chunk['hour'] = (((pd.to_datetime(chunk.charttime) - pd.to_datetime(first.charttime)).dt.total_seconds()) / 3600).round(0).astype(int)
    
    chunk = chunk.drop(columns=['stay_id', 'charttime'])
    chunk = chunk[chunk.hour < 24]
    
    chunk.value = pd.to_numeric(chunk.value)
    chunk = chunk[chunk.value < 400.0]
    chunk = chunk[chunk.value > 0.0]

    chunk = chunk.groupby('hour', as_index=False)['value'].mean()
    chunk = chunk.set_index('hour')

    if not chunk.empty:
        chunk_interpolated = pd.DataFrame({ 'hour' : range(0, 24, 1)})
        chunk_interpolated = chunk_interpolated.set_index('hour')
        
        chunk_interpolated = chunk_interpolated.reindex()
        chunk_interpolated.loc[chunk.index,'value'] = chunk.value

        chunk_interpolated = chunk_interpolated.interpolate().reset_index()
        chunk_interpolated['value'] = chunk_interpolated['value'].round(1)

        formatted_chunk = pd.DataFrame({'subject_id' : [first.index.values[0]]*24, 'value' : chunk_interpolated.value})
        formatted_chunk = formatted_chunk.set_index('subject_id')

        return formatted_chunk

    formatted_chunk = pd.DataFrame({'subject_id' : [first.index.values[0]]*24, 'value' : [np.nan]*24})
    formatted_chunk = formatted_chunk.set_index('subject_id')

    return formatted_chunk    

    first['value'] = np.nan
    return first

if __name__ == "__main__":
    print('Saving systollicbloodpressure feature...')
    main().to_pickle(output_root + 'sys_bp.pickle')
    print('Saved systollic blood pressures!\n')
