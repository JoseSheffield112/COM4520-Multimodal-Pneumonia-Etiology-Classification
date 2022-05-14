import numpy as np

def main(arr):
    arr = arr
    elements = 3
    n = 0
    s = np.zeros([elements,1])
    for row in arr:
        if type(row) is not type(np.nan):
            n += 1
            for i in range(0, elements):
                s[i] += row[i].astype(float)
    mean = np.array([(s[0]/n).astype(float), (s[1]/n).astype(float), (s[2]/n).astype(float)])

    sd = np.zeros([elements,1])
    for row in arr:
        if type(row) is not type(np.nan):
            for i in range(0, elements):
                sd[i] += (row[i].astype(float) - mean[i].astype(float))**2

    std = np.zeros([elements,1])
    for i in range(0, elements):
        std = (sd/n)**0.5

    out = []
    fill = np.zeros(elements) # Seems we're imputting 0's... could use np.array([mean[0],mean[1],mean[2]])

    for row in arr:
        #row = row.astype(np.float)
        if type(row) is type(np.nan):
            out.append(fill)
        else:
            row = row.astype(float) - mean[i].astype(float)
            row /= std[i].astype(float)
            out.append(row)
    return out