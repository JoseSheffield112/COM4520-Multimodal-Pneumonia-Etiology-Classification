# Run on your machine to ignore remote changes so that you don't get annoyed every time someone changes a file:
# git update-index --skip-worktree scripts/config.py

#CONFIGS YOU NEED TO PREPROCESS (A lot of these are relative paths, not all need to be tweaked)
save_npz = True
low_memory = False
save_intermediates = True
overwrite_cache = True

#TODO: Add some comments here explaining what each path is for.
in_data_root = 'C:/dev/darwin/datasets/mimic-iv-1.0'
cohort_root = 'cohorts'
out_data_root = 'datasets/mimic-iv/mimic-iv-full-cohort-v3'

image_data_root = 'datasets'
origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort-v3'
feature_script_root = 'scripts/features'
preprocessing_script_root = 'scripts/preprocessing'
intermediate_root = 'intermediates/'
feature_root = 'features'
labels_root = 'cohorts'

output_root = 'output'
#Statistics
#Relative path to root of folder where statistics are output
stats_root = 'stats'
#Relative path to root of folder where graphs are output
graphs_root = stats_root + '/graphs'
#Sql databse connection data 

#Path to json file that contains information needed to connect to a postgresql database
connection_json_root = 'config/connection.json'
# Root for the mimic-code repository. Needed to find some useful sql scripts for imputing features. 
mimic_code_root = 'C:/dev/darwin/mimic-code'


#CONFIGS YOU NEED TO RUN THE MODELS:
pretrained_root = 'pretrained'
impkPath = 'C:\dev\darwin\datasetExploration\data\im_w_icd10_updated.pk'