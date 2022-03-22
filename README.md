# COM4520---PreProcessing
MIMIC-iv pre-processing code

# Setup
- Place MIMIC-iv datasets in the `/datasets` folder
- Place cohorts in the `/cohorts` folder
- Be sure to have [Python 3](https://www.python.org/downloads/) installed

# Usage

## Generating a cohort dataset from the complete dataset and a cohort
1. Place the complete dataset inside the `/datasets` folder
2. Place the cohort folder inside the `/cohorts` folder
5. Modify `in_data_root`, `out_data_root`, `cohort_root` in `/scripts/config.py` to point to the respective folders
6. Run `/scripts/build-tables.py`

## Reduce or strip an existing dataset
1. Place the dataset inside the `/datasets` folder
2. Modify `origin_root`, `output_root` in `/scripts/config.py` to point to the dataset
4. Adjust the `build_tables` call in `/scripts/build-tables.py` to your liking
5. Run `/scripts/build-tables.py`

## Generating a feature from an existing dataset
1. Place the dataset inside the `/datasets` folder
2. Modify `origin_root` in `scripts/config.py` to point to the dataset
3. Run the corresponding script from `/scripts/features`
4. Find the output csv file in `/intermediates` and npy file in `/features`

## Generating every feature from an existing dataset
1. Place the dataset inside the `/datasets` folder
2. Modify `origin_root` in `scripts/config.py` to point to the dataset
3. Run `/scripts/build-features.py`
4. Find the output csv files in `/intermediates` and npy files in `/features`
5. Find the compiled output npz file in `/output`

# Features implemented
| Feature | Description | Value range | Values per patient |
| --- |---| --- | --- |
| AIDS | Whether the patient has Acquired ImmunoDeficiency Syndrome | x ∈ {0, 1} | 1 |
| Heartrate | The patients' hourly heartrate over 24h | 0 < x < 300 | 24
| MSCancer | Whether the patient has metastatic cancer | x ∈ {0, 1} | 1 |
| Temperature | The patients' hourly temperature (°C) over 24h | 25 < x < 50 | 24
| Whitebloodcells | The patient's white blood cell count | 0 < x < 1000 | ?

# Using this repository with low memory
The dataset we are working with is quite large so you will have trouble processing it if you are working on a machine with low memory. There are several ways we have allowed the code to be run with lower memory usage, but on some machines it will still be necessary to use a [reduced or stripped](#reduce-or-strip-an-existing-dataset) version of MIMIC-iv rather than using the original. A [cohort](#generating-a-cohort-dataset-from-the-complete-dataset-and-a-cohort) is an example of a reduced dataset.
- When reading through tables, we split the tables into chunks to be read one by one, dropping unneeded records and columns as soon as possible to make sure as little memory as possible is wasted.
- We implemented a low memory option to `scripts/build_features` which will allow the compiled npz output to be built from individual npy feature files, rather than building it in memory alongside the features.