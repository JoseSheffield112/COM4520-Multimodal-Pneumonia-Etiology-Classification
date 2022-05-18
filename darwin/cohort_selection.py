# Import libraries
import numpy as np
import pandas as pd
from darwin.config import *
import torchxrayvision as xrv
from sklearn.model_selection import train_test_split
from scipy import stats

def map_hosp_etiology(dataframe):
    hosp_df = pd.DataFrame(dataframe['hadm_id'].unique(), columns=['hadm_id'])
    hosp_df['etiology'] = np.nan
    for index, row in dataframe.sort_values(by=['seq_num']).iterrows():
        if row['hadm_id'] in hosp_df['hadm_id'].values:
            if hosp_df.loc[hosp_df['hadm_id'] == row['hadm_id']]['etiology'].isna().any():
                hosp_df.loc[hosp_df['hadm_id'] == row['hadm_id'], ['etiology']] = row['icd_type']
    return hosp_df


def classify_icd(icd_code):
    if str(icd_code) in pneumo_viral_icds or str(icd_code) in pneumo_viral_icds_10:
        return 1
    elif str(icd_code) in pneumo_bacterial_icds:
        return 2
    elif str(icd_code) == pneumo_unknown_icd:
        return 3
    else:
        return 0


def map_icd_9_code(patients_list, print_stats=False, icd_version_10=True):
    """
    Perform Bacterial/Viral classification based on ICD-9 codes
    :param icd_code:
    :return:
    """
    diagnoses = pd.read_csv(diagnoses_path)
    filtered_diagnoses = diagnoses[diagnoses.subject_id.isin(patients_list)]
    if icd_version_10:
        print('version 10 in process')
        filtered_diagnoses_9 = filtered_diagnoses
        filtered_diagnoses_9['icd_type'] = filtered_diagnoses_9['icd_code'].apply(classify_icd)
    else:
        filtered_diagnoses_9 = filtered_diagnoses[filtered_diagnoses.icd_version == 9]
        filtered_diagnoses_9['icd_type'] = filtered_diagnoses_9['icd_code'].apply(classify_icd)

    vir_icd = filtered_diagnoses_9[(filtered_diagnoses_9.icd_type == 1)]
    bac_icd = filtered_diagnoses_9[(filtered_diagnoses_9.icd_type == 2)]
    held_out_idiopathic_icd = filtered_diagnoses_9[(filtered_diagnoses_9.icd_type == 3)]

    labeled_vir = map_hosp_etiology(vir_icd)
    labeled_bac = map_hosp_etiology(bac_icd)

    if print_stats:
        print('Number of unique patients of Bac w/ CXR', len(bac_icd['subject_id'].unique()))
        print('Number of unique hospitalisations of Bac w/ CXR', len(bac_icd['hadm_id'].unique()))
        print('Number of unique patients of Vir w/ CXR', len(vir_icd['subject_id'].unique()))
        print('Number of unique hospitalisations of Vir w/ CXR', len(vir_icd['hadm_id'].unique()))
        print('Number of unique patients of Idio Pneumonia w/ CXR', len(held_out_idiopathic_icd['subject_id'].unique()))
        print('Number of unique hospitalisations of Idio Pneumonia w/ CXR',
              len(held_out_idiopathic_icd['hadm_id'].unique()))

    return labeled_vir, labeled_bac


def map_hosp_to_study(study_date, admit_time, disch_time):
    """
    :param patient_id:
    :param study_date:
    :return:
    """
    if int(admit_time) <= int(study_date) <= int(disch_time):
        return True
    else:
        return False


def change_date_format(my_string):
    x = str(my_string).split()
    res = x[0].replace('-', '')
    return res


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def get_pneumonia_cohort(csv_df, meta_df):
    # Get Patients for which Pneumonia was detected
    csv_df = csv_df[['subject_id', 'study_id', 'Pneumonia']]
    pneumonia_df = csv_df[(csv_df['Pneumonia'].isin([1.0, -1.0]))]
    unique_studyIds = meta_df.drop_duplicates(
        subset=['subject_id', 'study_id', 'ViewPosition'])
    filtered_studyDates = pneumonia_df.merge(unique_studyIds, on=['study_id', 'subject_id'], how='left')
    filtered_studyDates = filtered_studyDates[(filtered_studyDates["ViewPosition"].isin(["AP"]))]

    # Get only adults patients
    uniq_patients_list = filtered_studyDates['subject_id'].unique()
    patients = pd.read_csv(patients_path)
    adults = patients.drop(patients[patients.anchor_age < 15].index)
    patient_and_adult = adults.drop(adults[~adults.subject_id.isin(uniq_patients_list)].index)
    
    # Map icd-9 codes to etiology (bacterial/viral) w.r.t 'seq_num' (ranking of diagnoses)
    labeled_vir, labeled_bac = map_icd_9_code(patient_and_adult['subject_id'], print_stats=True, icd_version_10=False)

    # Map admission back to patient
    admissions = pd.read_csv(admissions_path)
    vir_hosp_patient = labeled_vir.merge(admissions, how='left', on='hadm_id').filter(
        items=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'etiology'])
    bac_hosp_patient = labeled_bac.merge(admissions, how='left', on='hadm_id').filter(
        items=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'etiology'])

    # Change the date formats and merge with pneumonia
    vir_hosp_patient['admittime'] = vir_hosp_patient['admittime'].apply(change_date_format)
    vir_hosp_patient['dischtime'] = vir_hosp_patient['dischtime'].apply(change_date_format)
    bac_hosp_patient['admittime'] = bac_hosp_patient['admittime'].apply(change_date_format)
    bac_hosp_patient['dischtime'] = bac_hosp_patient['dischtime'].apply(change_date_format)

    final_vir = vir_hosp_patient.merge(filtered_studyDates, how='left', on="subject_id")
    final_bac = bac_hosp_patient.merge(filtered_studyDates, how='left', on="subject_id")

    final_vir['valid'] = final_vir.apply(lambda x: map_hosp_to_study(x['StudyDate'], x['admittime'], x['dischtime']),
                                         axis=1)
    final_bac['valid'] = final_bac.apply(lambda x: map_hosp_to_study(x['StudyDate'], x['admittime'], x['dischtime']),
                                         axis=1)

    viral_samples = final_vir[final_vir['valid'] == True]
    viral_samples.drop('valid', 1, inplace=True)

    bacterial_samples = final_bac[final_bac['valid'] == True]
    bacterial_samples.drop('valid', 1, inplace=True)

    pneumo_data = pd.concat([viral_samples, bacterial_samples])

    print('Data before split:', pneumo_data)
    pneumo_data.to_csv(pneumo_processed_full_path)

    return pneumo_data
