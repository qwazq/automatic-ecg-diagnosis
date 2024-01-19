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
rowN=0
personN=0

for row in predictOutput_np:
    rowN+=1
    for person_per in row:
        # personN+=1
        x_list.append(person_per)
        y_list.append(rowN)

rowN=0
personN=0

print(y_list)

for row in gold_standard_np:
    rowN+=1
    for person_per in row:
        personN+=1
        print(personN)
        if person_per==1:
            c_list.append("r")
            y_list[personN]+=0.3
        else:
            c_list.append("g")
#

# print(predictOutput_np)
# print(gold_standard_np.shape)

# print(c_list)

y = np.array(x_list)
x = np.array(y_list)
colors = np.array(c_list)

plt.scatter(x, y, c=colors,alpha=0.3,s=40)
plt.show()
