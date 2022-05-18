# Import libraries
from idna import valid_contextj
import numpy as np
import pandas as pd
from torch import rand
from darwin.config import *

def filter_rows_by_values(df, col1, col2, values1, values2):
    return df[~ (df[col1].isin(values1) & df[col2].isin(values2))]


def perform_split(df, random_state=42, export_to_csv=False, remove_duplicate_hadm=False):
    if remove_duplicate_hadm:
        df = df.drop_duplicates(['subject_id', 'hadm_id'], keep='last')
    class_counts = df.groupby(['etiology']).size().to_numpy() 
    full_count = np.amin(class_counts)# get the # of samples from the class with less samples
    print(full_count)
    train_count = int(round(full_count*0.625,0))
    print(train_count)
    test_count = int(round(full_count*0.25,0))
    print(test_count)
    valid_count = int(round(full_count*0.125,0))
    print(valid_count)
    # np.random.seed(random_state)
    df.iloc[np.random.permutation(len(df))]
    print(df.groupby('etiology', group_keys=False).count())
    train = df.groupby('etiology', group_keys=False).sample(n=train_count, random_state=random_state)
    print(len(train))
    train.iloc[np.random.permutation(len(train))]
    df_after_train = filter_rows_by_values(df, 'subject_id', 'hadm_id', train['subject_id'], train['hadm_id'])
    val = df_after_train.groupby('etiology', group_keys=False).sample(n=valid_count, random_state=random_state)
    print(len(val))
    val.iloc[np.random.permutation(len(val))]
    df_after_val = filter_rows_by_values(df_after_train, 'subject_id', 'hadm_id', val['subject_id'], val['hadm_id'])
    test = df_after_val.groupby('etiology', group_keys=False).sample(n=test_count, random_state=random_state)
    print(len(test))
    test.iloc[np.random.permutation(len(test))]

    if export_to_csv:
        train.to_csv(pneumo_processed_train_path)
        test.to_csv(pneumo_processed_test_path)
        val.to_csv(pneumo_processed_valid_path)
    return train, val, test
