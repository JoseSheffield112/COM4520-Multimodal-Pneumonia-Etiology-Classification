# COM4520---PreProcessing
MIMIC-iv pre-processing code

# Setup
- Place datasets in the `/datasets`/ folder
- Place cohorts in the `/cohorts` folder

# Usage

## Generating a cohort dataset from the complete dataset and a cohort
1. Place the complete dataset inside the `/datasets` folder
2. Place the cohort folder inside the `/cohorts` folder
3. Modify `origin_root` in `/scripts/build-tables.py` to point to the complete dataset
4. Modify `output_root` in `/scripts/build-tables.py` to point to a new location
5. Modify `cohort_root` in `/scripts/build-tables.py` to point to the cohort folder
6. Run `/scripts/build-tables.py`

## Generating features from an existing dataset
1. Place the dataset inside the `/datasets` folder
2. Modify every `origin_root` in the scripts found in `/scripts/features` to point to the dataset
3. Run the scripts in `/scripts/features`
4. Find the output csv files in `/intermediates` and npy files in `/features`

## Reduce or strip an existing dataset
1. Place the dataset inside the `/datasets` folder
2. Modify `origin_root` in `/scripts/build-tables.py` to point to the dataset
3. Modify `output_root` in `/scripts/build-tables.py` to point to a new location
4. Adjust the `build_tables` call in `/scripts/build-tables.py` to your liking
5. Run `/scripts/build-tables.py`