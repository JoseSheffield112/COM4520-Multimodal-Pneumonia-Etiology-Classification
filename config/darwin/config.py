# Here you configure various aspects of the codebase.


########### (l) Data you need to configure ####################################################################################


#Regarding running the models:
# Path to the data required to run the models. This is what is output by running scripts/darwin/build-features.py.
dataPath = 'C:\dev\darwin\datasetExploration\data\ourim9April_img_squeezed.pk'

# Regarding the preprocessing:
# Point this to the root of where you store your mimic-iv csv data. 
root_mimiciv = 'C:/dev/darwin/datasets/mimic-iv-1.0'

# This is regarding the preprocessing of each individual feature. Set this to false if you want to use previous results from that step. Must be set to true for the first run.
overwrite_cache = True
# Mostly for debugging. Saves intermediate data from processing features.
save_intermediates = True



########### (ll) Data you likely don't need to configure. It mostly contains relative paths ####################################################################################

# Regarding running the models:
# Setting this is unnecessary for the current version of the paper (14th of May). This is the root folder for where pretrained weights were found for the image model.
pretrained_root = 'data/pretrained'

# Regarding the preprocessing:

cohort_root = 'data/cohorts'
out_stripped_mimic_root = 'data/datasets/mimic-iv/mimic-iv-full-cohort-v3'
# This is the root path to the mimic data that the preprocessing scripts will use. 
origin_root = out_stripped_mimic_root

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
stats_root = 'stats'
#Path to root of folder where graphs are output by default.
graphs_root = stats_root + '/graphs'


#Sql databse connection data 
#Path to json file that contains information needed to connect to a postgresql database. This was used for features that ended up being dropped. It's unnecessary to set this now.
connection_json_root = 'config/connection.json'
# Root for the mimic-code repository. https://github.com/MIT-LCP/mimic-code . This is now legacy and unnecessary. Some sql scripts were used for features that ended up being dropped.  
mimic_code_root = 'C:/dev/darwin/mimic-code'
