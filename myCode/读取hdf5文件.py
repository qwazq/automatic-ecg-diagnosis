import h5py
import numpy as np

hdf5FileWhere=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data\ecg_tracings.hdf5"

with h5py.File(hdf5FileWhere, "r") as f:
    x = np.array(f['tracings'])

print(x)