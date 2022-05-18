# Here you configure various aspects of the codebase.


########### (l) Data you need to configure ####################################################################################


#Regarding running the models:
# Path to the data required to run the models. This is what is output by running scripts/darwin/build-features.py.
dataPath = 'data/pretrained/ourim9April_img_squeezed.pk'

# Regarding the preprocessing:
# Point this to the root of where you store your mimic-iv csv data. 
root_mimiciv = 'data/datasets/mimic-iv-1.0'
# Point this to the folder containing the MIMIC-CXR-JPG image jpg files.
external_images_path = 'E:/files'

# This is regarding the preprocessing of each individual feature. Set this to false if you want to use previous results from that step. Must be set to true for the first run.
overwrite_cache = True
# Mostly for debugging. Saves intermediate data from processing features.
save_intermediates = True


########### (ll) Data you likely don't need to configure. These are mostly relative paths to the root of this repo ####################################################################################

# Regarding running the models:
# Setting this is unnecessary for the current version of the paper (14th of May). 
# This is the root folder for where pretrained weights are found for the image model.
pretrained_root = 'data/pretrained'

# Regarding the preprocessing:

cohort_root = 'data/cohorts'
out_stripped_mimic_root = 'data/datasets/mimic-iv/mimic-iv-full-cohort-v3'
# This is the root path to the mimic data that the preprocessing scripts will use. 
origin_root = out_stripped_mimic_root

# This is the path to the CXR dataset's csv files.
cxr_jpg_root = 'data/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/'

# Raw CXR related csv files (supplied as gz format)
cxr_negbio_csv = cxr_jpg_root + 'mimic-cxr-2.0.0-negbio.csv.gz'
cxr_metadata_csv = cxr_jpg_root + 'mimic-cxr-2.0.0-metadata.csv.gz'
cxr_chexpert_csv = cxr_jpg_root + 'mimic-cxr-2.0.0-chexpert.csv.gz'

# Preprocessed CSV files that will be generated.
pneumo_processed_full_path = cohort_root + '/before_split.csv'
pneumo_processed_train_path = cohort_root + '/train.csv'
pneumo_processed_test_path = cohort_root + '/test.csv'
pneumo_processed_valid_path = cohort_root + '/val.csv'

# Diagnoses csv file for getting ICD-9 codes froom
diagnoses_path = root_mimiciv + '/hosp/diagnoses_icd.csv.gz'

# Patient csv file for Age
patients_path = root_mimiciv + '/core/patients.csv.gz'

# Admission csv file for time comparison
admissions_path = root_mimiciv + '/core/admissions.csv.gz'

# Bank of ICD-9 codes for pneumonia.
pneumo_viral_icds = ['480', '4800', '4801', '4802', '4803', '4808', '4809', '4870', '48881', '48811', '48801', '4841']
pneumo_bacterial_icds = ['4845', '481', '482', '4820', '4821', '4822', '4823', '48230', '48231', '48232', '48239', '4824', '48240', '48241', '48242', '48249', '4828', '48281', '48282', '48283', '48284', '48289', '4829']
pneumo_unknown_icd = '486'
pneumo_viral_icds_10 = ['J120', 'J121', 'J122', 'J123', 'J128', 'J1281', 'J1281', 'J1282', 'J1289', 'J129']

# Path to where the image data for this cohort is held. Here it is stored as a serialized dictionary containingt the hadm_id of the related admission and the image data as a 3dimensional list.
image_data_pickled_root = 'data/datasets'

feature_script_root = 'scripts/darwin/features'
preprocessing_script_root = 'scripts/darwin/preprocessing'
intermediate_root = 'data/intermediates/'
feature_root = 'data/features'
labels_root = 'data/cohorts'
output_root = 'output'


#Statistics
#Path to root of folder where statistics are output when running the models by default.
stats_root = output_root + '/stats'
#Path to root of folder where graphs are output by default.
graphs_root = stats_root + '/graphs'


#Sql databse connection data 
#Path to json file that contains information needed to connect to a postgresql database. This was used for features that ended up being dropped. It's unnecessary to set this now.
connection_json_root = 'config/connection.json'
# Root for the mimic-code repository. https://github.com/MIT-LCP/mimic-code . This is now legacy and unnecessary. Some sql scripts were used for features that ended up being dropped.  
mimic_code_root = 'C:/dev/darwin/mimic-code'
