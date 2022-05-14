# Multimodal-based Pneumonia Etiology Classification codebase

# Dependencies
- Pytorch (LTS, 1.8.2) https://pytorch.org/get-started/locally/
- TorchXrayVision https://github.com/mlmed/torchxrayvision

# Obtaining the data to run the models
This assumes that you have access to mimic-iv https://mimic.mit.edu/ .

1. Within the `config/darwin/config.py` script, set the value of root_mimiciv to the path of the root folder of where on your computer you hold the mimic iv csv data.
2. Run `scripts/darwin/build-tables.py` to get a stripped version of mimic-iv.
3. Run `scripts/darwin/build-features.py` to get the data for the models. This will be stored under 'output/im.pk' .

# Running the models

1. Within the `config/darwin/config.py` script, set the value of the `dataPath` variable to be the path of the output of the previous scripts/darwin/build-features.py script. That is, the `output/im.pk` file.

Use the scripts/darwin/experiment.py to run the models. From the root folder of this repo, run `python scripts/darwin/experiment.py -h` for guidance on how to run an experiment.

Example of a possible experiment:
`python scripts\experiment.py -m image_static -sf True -nr 20 -ne 20 -o [pathToRootOfExperiment] -en last_experiment_image_static_1 -estp True -aug False --earlyStopMetric valid -pat 7`

Will run the multimodal image and static model for 20 runs and 20 epochs, with a random train/test split for every run with data augmentation enabled (Every time an image sample is retreived from the train set, it is randomly scaled and rotated to help avoid regularization). Early stop is enabled, using validation loss as a metric with patience=7 (If valloss does not improve after 7 runs, the training will stop).

# Features implemented
| Feature | Description | Range of possible values | Values per admission |
| --- |---| --- | --- |
| AIDS | Whether the patient has Acquired ImmunoDeficiency Syndrome | x ∈ {0, 1} | 1 |
| Gender | Whether the patient is male or female | x ∈ {0, 1} | 1 |
| Influenza | Whether the patient has the flu | x ∈ {0, 1} | 1 |
| Heartrate | The patients' hourly heartrate over 24h | 0 < x < 300 | 24
| Hematocrit | The patient's red blood cell count min/max/mean | ? < x < ? | 3
| MSCancer | Whether the patient has metastatic cancer | x ∈ {0, 1} | 1 |
| Mycoplasma | Whether the patient has mycoplasma pneumoniae | x ∈ {0, 1} | 1 |
| PO2FO2Ratio | The ratio of arterial oxygen to inspired oxygen over 24h | ? < x < ? | 24
| RSV | Whether the patient has Respiratory Syncytial Virus | x ∈ {0, 1} | 1 |
| SARS | Whether the patient has SARS-CoV | x ∈ {0, 1} | 1 |
| Staphylococcus | Whether the patient has Staphylococcus | x ∈ {0, 1} | 1 |
| Systolic_Blood_Pressure | The patients' hourly systolic blood pressure over 24h | ? < x < ? | 24
| Temperature | The patients' hourly temperature (°C) over 24h | 25 < x < 50 | 24
| Whitebloodcells | The patient's white blood cell count min/max/mean | 0 < x < 1000 | 3

# Using this repository with low memory
The dataset we are working with is quite large so you will have trouble processing it if you are working on a machine with low memory. There are several ways we have allowed the code to be run with lower memory usage, but on some machines it will still be necessary to use a [reduced or stripped](#reduce-or-strip-an-existing-dataset) version of MIMIC-iv rather than using the original. A [cohort](#generating-a-cohort-dataset-from-the-complete-dataset-and-a-cohort) is an example of a reduced dataset.
- When reading through tables, we split the tables into chunks to be read one by one, dropping unneeded records and columns as soon as possible to make sure as little memory as possible is wasted.
- ~~We implemented a low memory option to `scripts/build_features` which will allow the compiled npz output to be built from individual npy feature files, rather than building it in memory alongside the features.~~
