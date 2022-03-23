# Run on your machine to ignore remote changes so that you don't get annoyed every time someone changes a file:
# git update-index --skip-worktree scripts/config.py

save_npz = True
low_memory = False
save_intermediates = True
in_data_root = 'datasets/mimic-iv/mimic-iv-full-cohort-v2'
out_data_root = 'datasets/mimic-iv/mimic-iv-full-cohort-v3'
cohort_root = 'cohorts/data_oscar_cohort'
origin_root = 'datasets/mimic-iv/mimic-iv-full-cohort-v2'
feature_script_root = 'scripts/features'
intermediate_root = 'intermediates/'
feature_root = 'features'
output_root = 'output'


#Sql databse connection data 
#Path to json file that contains information needed to connect to a postgresql database
connection_json_root = 'config/connection.json'

# Root for the mimic-code repository. Needed to find some useful sql scripts for imputing features. 
mimic_code_root = 'C:/dev/darwin/mimic-code'
