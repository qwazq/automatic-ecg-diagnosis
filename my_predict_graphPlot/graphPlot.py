import matplotlib.pyplot as plt
import numpy as np
#
from myFunctionFolder.my_CSV_Function import getCsvReader, CsvTo2DList
from myFunctionFolder.my_Numpy_Function import *

#
x_list=[]
y_list=[]
c_list=[]

#
predictOutput_path="./predictOutput.csv"
predictOutput_np=csvToNp(predictOutput_path)

gold_standard_path="gold_standard.csv"
gold_standard_np=csvToNp(gold_standard_path)


#
predictOutput_np=predictOutput_np.T
gold_standard_np=gold_standard_np.T

#

print(predictOutput_np)
# print(gold_standard_np.shape)

