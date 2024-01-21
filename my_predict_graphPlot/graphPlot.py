import matplotlib.pyplot as plt
import numpy as np
#
from myFunctionFolder.my_CSV_Function import getCsvReader, CsvTo2DList, list2DWriteTo
from myFunctionFolder.my_Numpy_Function import *

#
predictOutput_path="./predictOutput.csv"
predictOutput_np=csvToNp(predictOutput_path)

gold_standard_path="gold_standard.csv"
gold_standard_np=csvToNp(gold_standard_path)

#


xyc_list=np.zeros((6,predictOutput_np.shape[0],4), dtype = "float", order = 'C')




pass

#
'''
至此，csv被转置为横版
'''
predictOutput_np=predictOutput_np.T
gold_standard_np=gold_standard_np.T

#
N_row=0
N_col=0
for row in predictOutput_np:
    for count in row:
        xyc_list[N_row][N_col][0]=N_row
        xyc_list[N_row][N_col][1]=N_col
        xyc_list[N_row][N_col][2]=count
        xyc_list[N_row][N_col][3]=0

        N_col += 1
    pass
    N_col=0
    N_row += 1
pass

#
# N_row=0
# N_col=0
# for row in gold_standard_np:
#     for count in row:
#
#         if count:
#             xyc_list[N_row][N_col][3]="r"
#         else:
#             xyc_list[N_row][N_col][3] = "g"
#
#         N_col += 1
#     pass
#     N_col=0
#     N_row += 1
# pass
#
xyc_list_output_list= "xyc_list.csv"
list2DWriteTo(xyc_list[0],xyc_list_output_list)
#
# N_row=0
# N_col=0
# for row in predictOutput_np:
#     for count in row:
#
#
#     pass
#     N_col=0
#     N_row += 1
# pass
# xyc_list_output_list= "xyc_list"
# # # list2DWriteTo(xyc_list[0],xyc_list_0_path)
# # # osOpenFile(xyc_list_0_path)
# #
# write_3DnpTo2DCsv(xyc_list, xyc_list_output_list, 1)

#

# print(predictOutput_np)
# print(gold_standard_np.shape)

# x=
#
# # plt.scatter(x, y, c=colors,alpha=0.3,s=40)
# print(xyc_list[0,:,1])
plt.scatter(xyc_list[0,:,0],xyc_list[0,:,2])
plt.show()
pass