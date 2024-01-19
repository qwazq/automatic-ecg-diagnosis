import os

import numpy as np

from myFunctionFolder.my_Numpy_Function import np2DWriterToCsv
from myFunctionFolder.my_OS_Function import osOpen

# pwd=os.path[0]

outputFile="./predictOutput.npy"

outputList=np.load(outputFile)

np2DWriterToCsv("predictOutput.csv",outputList)

osOpen("predictOutput.csv")