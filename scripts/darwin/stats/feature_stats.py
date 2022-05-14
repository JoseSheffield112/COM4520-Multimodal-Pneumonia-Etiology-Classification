import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from config.darwin.config import *
from os import listdir

def get_features(target):
    cohort = pd.read_csv(labels_root + '/' + target + '.csv', header=0)
    features = {}
    failed = {}
    print('Feature | Admissions present')
    for filename in listdir(feature_script_root):
        if filename.endswith('.py'):
            file = filename[:-3] # No extension
            try:
                features[file] = pd.read_pickle(feature_root + '/' + file + '.pickle')
                print(file, features[file][features[file].index.isin(cohort.hadm_id.values)].shape[0], sep=' | ')
            except Exception as e:
                failed[file] = e
                print(file, '-', sep=' | ')
    print()
    return [features[f] for f in ['age', 'gender', 'aids', 'mscancer', 'mycoplasma', 'staphylococcus']]

def generate_subset_stats(target):
    features = get_features(target)

    print('Statistic | Viral | Bacterial | Both (conflicting etiologies) | Total')
    cohort = pd.read_csv(labels_root + '/' + target + '.csv', header=0)

    viral = cohort[cohort.etiology==1.0]
    bacterial = cohort[cohort.etiology==2.0]

    conflicts = pd.read_csv(labels_root + '/' + target + '.csv', index_col=[0], usecols=['hadm_id', 'etiology'], header=0)
    conflicts = conflicts.groupby(level=0).mean()
    conflicts = conflicts[(conflicts.etiology != 1.0) & (conflicts.etiology != 2.0)]
    csv_path = Path(output_root + '/conflicts.csv')  
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if target == 'data':
        conflicts.to_csv(csv_path)

    print('Study count', len(viral), len(bacterial), '0', len(cohort), sep=' | ')
    print('Admission count', len(viral.hadm_id.unique()), len(bacterial.hadm_id.unique()), len(conflicts.index.unique()), len(cohort.hadm_id.unique()), sep=' | ')

    cohort = cohort[~cohort.subject_id.duplicated(keep='first')].drop(columns='subject_id').set_index('hadm_id')
    viral = viral[~viral.subject_id.duplicated(keep='first')]
    bacterial = bacterial[~bacterial.subject_id.duplicated(keep='first')]
    conflicted = viral[viral.subject_id.isin(bacterial.subject_id.unique())]

    viral = viral.drop(columns='subject_id').set_index('hadm_id')
    bacterial = bacterial.drop(columns='subject_id').set_index('hadm_id')
    conflicted = conflicted.drop(columns='subject_id').set_index('hadm_id')
    print('Patient count', len(viral), len(bacterial), len(conflicted), len(cohort), sep=' | ')

    age = features[0]
    viral_age = pd.merge(age, viral, left_index=True, right_index=True)
    bacterial_age = pd.merge(age, bacterial, left_index=True, right_index=True)
    conflicted_age = pd.merge(age, conflicted, left_index=True, right_index=True)
    print('Age mean', round(np.mean(viral_age.age), 1), round(np.mean(bacterial_age.age), 1), round(np.mean(conflicted_age.age), 1), round(np.mean(age.age), 1), sep=' | ')
    print('Age std', round(np.std(viral_age.age), 1), round(np.std(bacterial_age.age), 1), round(np.std(conflicted_age.age), 1), round(np.std(age.age), 1), sep=' | ')

    gender = features[1]
    males = gender[gender.gender==0]
    males = pd.merge(males, cohort, left_index=True, right_index=True)
    females = gender[gender.gender==1]
    females = pd.merge(females, cohort, left_index=True, right_index=True)
    viral_males = pd.merge(males, viral, left_index=True, right_index=True)
    viral_females = pd.merge(females, viral, left_index=True, right_index=True)
    bacterial_males = pd.merge(males, bacterial, left_index=True, right_index=True)
    bacterial_females = pd.merge(females, bacterial, left_index=True, right_index=True)
    conflicted_males = pd.merge(males, conflicted, left_index=True, right_index=True) # Looking at the results table, we can deduce that this df is missing four records because reasons.
    conflicted_females = pd.merge(females, conflicted, left_index=True, right_index=True)
    print('Male count', len(viral_males), len(bacterial_males), len(conflicted_males), len(males), sep=' | ')
    print('Female count', len(viral_females), len(bacterial_females), len(conflicted_females), len(females), sep=' | ')

    for f in range(2, 6):
        label = ['aids', 'mscancer', 'mycoplasma', 'staphylococcus'][f-2]
        feature = features[f]
        pos = feature[feature[label]==1.0]
        pos = pd.merge(pos, cohort, left_index=True, right_index=True)
        viral_pos = pd.merge(pos, viral, left_index=True, right_index=True)
        bacterial_pos = pd.merge(pos, bacterial, left_index=True, right_index=True)
        conflicted_pos = pd.merge(pos, conflicted, left_index=True, right_index=True)
        print(label + ' count', len(viral_pos), len(bacterial_pos), len(conflicted_pos), len(pos), sep=' | ')

if __name__=='__main__':
    #for t in ['data', 'test', 'train', 'valid']:
    for t in ['unique_patients', 'bacterial_patients', 'viral_patients']:
        print('\n' + t.upper())
        generate_subset_stats(t)