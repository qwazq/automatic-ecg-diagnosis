import os

import numpy as np

from myFunctionFolder.my_CSV_Function import np2DToCsv
from myFunctionFolder.my_OS_Function import osOpen

# pwd=os.path[0]

outputFile="./predictOutput.npy"

outputList=np.load(outputFile)

np2DToCsv("predictOutput.csv", outputList)

osOpen("predictOutput.csv")