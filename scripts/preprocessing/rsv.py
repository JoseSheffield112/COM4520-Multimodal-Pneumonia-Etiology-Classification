import numpy as np

def main(arr):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    arr = np.nan_to_num(arr, nan=mean)
    arr = arr - mean
    arr = arr / std
    return arr