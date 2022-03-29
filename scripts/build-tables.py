import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import numpy as np
import pandas as pd
from pathlib import Path
from scripts.config import *

def main():
    build_tables('full', cohort=True)

def build_tables(mode='full', sample=None, cohort=False):
    '''
    Builds reduced or stripped MIMIC-IV.
    Optional keyword arguments:
    mode:       string describing which columns to include, default full
                    'full'      :   Include all columns.
                    'fsa'       :   Exclude columns irrelevant to Feature Set A in the MIMIC-III preprocessing code.
                    'minimal'   :   Exclude columns irrelevant to Feature Set A.
    icd_dict:   dictionary of accepted icd9 codes for each table
    sample:     the number of adult patients to include, default all
    cohort:     if True, build from list of patients in cohort_root/data.csv
    '''

    icd_dict = get_icd_dict()
    column_dict = get_column_dict(mode)

    subject_ids = build_patients(column_dict['patients'], sample, cohort)
    hadm_ids = build_admissions(subject_ids, column_dict['admissions'])
    stay_ids = build_icustays(subject_ids, hadm_ids, column_dict['icustays'])
    build_diagnoses_icd(subject_ids, hadm_ids, column_dict['diagnoses_icd'])
    build_labevents(subject_ids, hadm_ids, column_dict['labevents'], icd_dict['labevents']) # 12.8 GB input
    build_services(subject_ids, hadm_ids, column_dict['services'])
    build_chartevents(subject_ids, hadm_ids, stay_ids, column_dict['chartevents'], icd_dict['chartevents']) # 27.7 GB input
    build_outputevents(subject_ids, hadm_ids, stay_ids, column_dict['outputevents'], icd_dict['outputevents'])

def get_column_dict(mode):
    return {
        'full': {
            'patients'      : ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod'],
            'admissions'    : ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status', 'ethnicity', 'edregtime', 'edouttime', 'hospital_expire_flag'],
            'icustays'      : ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los'],
            'diagnoses_icd' : ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'],
            'labevents'     : ['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments'],
            'services'      : ['subject_id', 'hadm_id', 'transfertime', 'prev_service', 'curr_service'],
            'chartevents'   : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'storetime', 'itemid', 'value', 'valuenum', 'valueuom', 'warning'],
            'outputevents'  : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'storetime', 'itemid', 'value', 'valueuom']
        },
        'fsa': {
            'patients'      : ['subject_id', 'anchor_age', 'anchor_year'],
            'admissions'    : ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type'],
            'icustays'      : ['subject_id', 'hadm_id', 'stay_id', 'intime'],
            'diagnoses_icd' : ['subject_id', 'hadm_id', 'icd_code', 'icd_version'],
            'labevents'     : ['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'],
            'services'      : ['subject_id', 'hadm_id', 'transfertime', 'curr_service'],
            'chartevents'   : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom'],
            'outputevents'  : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valueuom'],
        },
        'minimal': {
            'patients'      : ['subject_id', 'anchor_age', 'anchor_year'],
            'admissions'    : ['subject_id', 'hadm_id', 'admittime', 'admission_type'],
            'icustays'      : ['subject_id', 'hadm_id', 'stay_id', 'intime'],
            'diagnoses_icd' : ['subject_id', 'hadm_id', 'icd_code', 'icd_version'],
            'labevents'     : ['labevent_id', 'subject_id', 'hadm_id', 'itemid', 'charttime', 'value'],
            'services'      : ['subject_id', 'hadm_id', 'transfertime', 'curr_service'],
            'chartevents'   : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value'],
            'outputevents'  : ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value'],
        }
    }[mode]

def get_icd_dict():
    return {
        'labevents'     : [11100, 11180, 50821, 50816, 51006, 51300, 51301, 50882, 950824, 50983, 50822, 50971, 50885, 50810, 51221],
        'chartevents'   : [723, 454, 184, 223900, 223901, 220739, 51, 442, 455, 6701, 220179, 220050, 211, 220045, 678, 223761, 676, 223762, 223835, 3420, 3422, 190],
        'outputevents'  : [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40056, 40056, 40405, 40428, 40086, 40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
    }

def build_patients(columns=None, samplesize=None, cohort=False):
    print('Building patients...')
    iter_csv = pd.read_csv(in_data_root + '/core/patients.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    data = pd.concat([chunk[chunk.anchor_age > 15] for chunk in iter_csv])
    print('Read origin')
    if cohort:
        print('Reading cohort...')
        cohort = pd.read_csv(cohort_root + '/data.csv', header=0, index_col=[0])
        subject_ids = cohort.index.unique()
        data = data[data.index.isin(subject_ids)]
    if samplesize:
        data = data.head(samplesize)
        print("Reduced sample size to " + str(samplesize) + " records")
    output_path = Path(out_data_root + '/core/patients.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved patients!\n')
    return data.index.unique()

def build_admissions(subject_ids=None, columns=None):
    print('Building admissions...')
    iter_csv = pd.read_csv(in_data_root + '/core/admissions.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    if subject_ids.any():
        data = pd.concat([chunk[chunk.index.isin(subject_ids)] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    data = data[~data.index.duplicated(keep='first')]
    print('Filtered duplicates')
    output_path = Path(out_data_root + '/core/admissions.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved admissions!\n')
    return data['hadm_id'].unique()

def build_diagnoses_icd(subject_ids=None, hadm_ids=None, columns=None):
    print('Building diagnoses_icd...')
    iter_csv = pd.read_csv(in_data_root + '/hosp/diagnoses_icd.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    if hadm_ids.any():
        data = pd.concat([chunk[((chunk.index.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids))) & (chunk.icd_version==9)] for chunk in iter_csv])
    else:
        data = pd.concat([chunk[data.icd_version==9] for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/hosp/diagnoses_icd.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved diagnoses_icd!\n')

def build_labevents(subject_ids=None, hadm_ids=None, columns=None, itemid_filter=None):
    print('Building labevents...')
    iter_csv = pd.read_csv(in_data_root + '/hosp/labevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    if hadm_ids.any() and itemid_filter:
        data = pd.concat([chunk[((chunk.subject_id.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids))) & (chunk.itemid.isin(itemid_filter))] for chunk in iter_csv])
    elif hadm_ids.any():
        data = pd.concat([chunk[chunk.hadm_id.isin(hadm_ids)] for chunk in iter_csv])
    elif itemid_filter:
        data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/hosp/labevents.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved labevents!\n')
    return data.index.unique()

def build_services(subject_ids=None, hadm_ids=None, columns=None):
    print('Building services...')
    iter_csv = pd.read_csv(in_data_root + '/hosp/services.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    if hadm_ids.any():
        data = pd.concat([chunk[(chunk.index.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids))] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/hosp/services.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved services!\n')

def build_icustays(subject_ids=None, hadm_ids=None, columns=None):
    print('Building icustays...')
    iter_csv = pd.read_csv(in_data_root + '/icu/icustays.csv', header=0, index_col=[0], iterator=True, chunksize=1000, usecols=columns)
    if hadm_ids.any():
        data = pd.concat([chunk[(chunk.index.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids))] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/icu/icustays.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved icustays!\n')
    return data['stay_id'].unique()

def build_chartevents(subject_ids=None, hadm_ids=None, stay_ids=None, columns=None, itemid_filter=None):
    print('Building chartevents...')
    iter_csv = pd.read_csv(in_data_root + '/icu/chartevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    if stay_ids.any() and itemid_filter:
        data = pd.concat([chunk[((chunk.index.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids)) | (chunk.stay_id.isin(stay_ids))) & (chunk.itemid.isin(itemid_filter))] for chunk in iter_csv])
    elif stay_ids.any():
        data = pd.concat([chunk[chunk.stay_id.isin(stay_ids)] for chunk in iter_csv])
    elif itemid_filter:
        data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/icu/chartevents.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved chartevents!\n')

def build_outputevents(subject_ids=None, hadm_ids=None, stay_ids=None, columns=None, itemid_filter=None):
    print('Building outputevents...')
    iter_csv = pd.read_csv(in_data_root + '/icu/outputevents.csv', header=0, index_col=[0], iterator=True, chunksize=10000, usecols=columns)
    if stay_ids.any() and itemid_filter:
        data = pd.concat([chunk[((chunk.index.isin(subject_ids)) | (chunk.hadm_id.isin(hadm_ids)) | (chunk.stay_id.isin(stay_ids))) & (chunk.itemid.isin(itemid_filter))] for chunk in iter_csv])
    elif stay_ids.any():
        data = pd.concat([chunk[chunk.stay_id.isin(stay_ids)] for chunk in iter_csv])
    elif itemid_filter:
        data = pd.concat([chunk[chunk.itemid.isin(itemid_filter)] for chunk in iter_csv])
    else:
        data = pd.concat([chunk for chunk in iter_csv])
    print('Read origin')
    output_path = Path(out_data_root + '/icu/outputevents.csv')  
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    data.to_csv(output_path)
    print('Saved outputevents!\n')

if __name__ == '__main__':
    main()
