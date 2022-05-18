# Multimodal-based Pneumonia Etiology Classification codebase

# Dependencies
- Pytorch (LTS, 1.8.2) https://pytorch.org/get-started/locally/
- TorchXrayVision https://github.com/mlmed/torchxrayvision

# Obtaining the data to run the models
This assumes that you have access to mimic-iv https://mimic.mit.edu/.

If you want to also obtain the stripped version of mimic-iv:

1. Within the `darwin/config.py` script, set the value of root_mimiciv to the path of the root folder of where on your computer you have the mimic-iv data stored.
2. Run `scripts/darwin/build-tables.py` to get a stripped version of mimic-iv.
3. Run `darwin/cohort_pickler`. It will generate within the path pointed to by the `image_data_pickled_root` variable defined in `darwin/config.py`, three pk files named test.pk, train.pk and valid.pk. It will also generate before_split.csv, test.csv, train.csv and valid.csv files which contain data in the cohort within the path pointed to by the `cohort_root` variable inside `darwin/config.py`.
4. Run `scripts/darwin/build-features.py` to get the data for the models. This will be stored under `output/im.pk`.

Alternatively, if you don't want to obtain the stripped version of mimic-iv:
1. Within the `darwin/config.py` change the value of the `origin_root` variable to point to the root folder of where on your computer you have the mimic-iv data stored.
2. Run `darwin/cohort_pickler`. It will generate within the path pointed to by the `image_data_pickled_root` variable defined in `darwin/config.py`, three pk files named test.pk, train.pk and valid.pk. It will also generate before_split.csv, test.csv, train.csv and valid.csv files which contain data in the cohort within the path pointed to by the `cohort_root` variable inside `darwin/config.py`.
3. Run `darwin/build-features.py` to get the data for the models. This will be stored under `output/im.pk`.

NOTE TO TEAM: We'll make a batch script for windows users and a bash shell for linux users to run all these scripts sequentially. After Johoo's and Max's scripts are in place.

# Running the models

1. Within the `scripts/darwin/config.py` script, set the value of the `dataPath` variable to be the path of the output of the previous `scripts/darwin/build-features.py` script. That is, the `output/im.pk` file.

Use the `darwin/experiment.py` to run the models. From the root folder of this repo, run: 

```python darwin/experiment.py -h```

for guidance on how to run an experiment.

## Example of a possible experiment:

```python darwin/experiment.py -m image_static -sf True -nr 20 -ne 20 -o [pathToRootOfExperiment] -en [experimentName] -estp True -aug False --earlyStopMetric valid -pat 7```

Will run the multimodal image and static model for 20 runs and 20 epochs, with a random train/test split for every run with data augmentation enabled (Every time an image sample is retreived from the train set, it is randomly scaled and rotated to help avoid regularization). Early stop is enabled, using validation loss as a metric with patience=7 (If valloss does not improve after 7 runs, the training will stop). 

The results of the experiment will be stored at [pathToRootOfExperiment]\\[experimentName]\\[modelName]. modelName is the same value as what was input to the -m option.
