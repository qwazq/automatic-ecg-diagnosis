import matplotlib.pyplot as plt
import numpy as np
#
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_Numpy_Function import *

#
predictOutput_path = "./predictOutput.csv"
predictOutput_np = csvToNp(predictOutput_path)

gold_standard_path = "gold_standard.csv"
gold_standard_np = csvToNp(gold_standard_path)

#

# xyc_list_dtype = np.dtype([('sort', 'int'), ('id', 'int'), ('pre', 'float'), ('color', 'str')])
# xyc_list=np.zeros((predictOutput_np.shape[0]*6), dtype =xyc_list_dtype, order = 'C')
xyc_list=[]

pass

#
'''
至此，csv被转置为横版
'''
predictOutput_np = predictOutput_np.T
gold_standard_np = gold_standard_np.T

#
N_row = 0
N_col = 0
persionID=0
for row in predictOutput_np:
    for count in row:
        color = ""
        if gold_standard_np[N_row][N_col] == 1:
            color = "r"
        else:
            color = "g"

        xyc_list.append([N_row, N_col, count, color])

        N_col += 1
        persionID+=1
    pass
    N_col = 0
    N_row += 1

    if persionID ==100:break
pass

xyc_list=np.array(xyc_list)

xyc_list_output_path= "xyc_list.csv"
list2DToCsv(xyc_list, xyc_list_output_path, isOpen=0)
#

