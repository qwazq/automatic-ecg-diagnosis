import os

#导入所需的包
import numpy as np


from myFunctionFolder.my_Numpy_Function import *
from myFunctionFolder.my_OS_Function import *

#导入npy文件路径位置
test = np.load(r"./predictOutput.npy")
print(test.shape)

np2DWriterToCsv(r"./predictOutput.npy",r"./predictOutput.csv")


osOpen(r"./predictOutput.csv")
# print(test)