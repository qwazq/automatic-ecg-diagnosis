import h5py
import numpy as np

from myFunctionFolder.my_CSV_Function import getCsvWriter
from myFunctionFolder.my_Numpy_Function import np2DWriterToCsv, write_3DnpTo2DCsv
#
from myFunctionFolder.my_OS_Function import pathDownToByList, osOpenFile, joinPathAndName, joinToFileName

from myFunctionFolder.my_hdf5_Function import get_npAarray_from_hdf5

#
dataFolder = r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data"
hdf5FileName = r"ecg_tracings.hdf5"
hdf5FilePath = pathDownToByList(dataFolder, [hdf5FileName])

outputFolder=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_code\outputs"
# outputFile=r"ecg_tracings.csv"

groupName = "tracings"

nparry = get_npAarray_from_hdf5(hdf5FilePath, groupName)



write_3DnpTo2DCsv(nparry, outputFolder, isOpen=1)
