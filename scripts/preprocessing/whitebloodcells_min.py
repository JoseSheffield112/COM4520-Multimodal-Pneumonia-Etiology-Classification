import numpy as np

def main(arr):
    arr = arr
    n = 0
    s = 0
    for row in arr:
        if not np.isnan(row).all():
            n += 1
            s += np.nanmean(row.astype(float))
    mean = s/n

    sd = 0
    for row in arr:
        if not np.isnan(row).all():
            sd += (row.astype(float) - mean)**2

    std = (sd/n)**0.5

    out = []
    fill = np.zeros(24)

    for row in arr:
        if np.isnan(row).all():
            out.append(fill)
        else:
            row -= mean
            row /= std
            out.append(row)

    return out