#
from myFunctionFolder.my_CSV_Function import csvTo2DList
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_python_Function import *

#
path_xyc_list = "xyc_list.csv"
xyc_list = csvTo2DList(path_xyc_list)
# new_xyc_list = []
# pass
# print(xyc_list)
for row in xyc_list:
    row[0]=int(float(row[0]))
    row[1]=int(float(row[1]))
    row[2]=float(row[2])
    row[3]=int(float(row[3]))

#
pass
#
xyc_list=list2DTolistTupe1D(xyc_list)

xyc_dtype = np.dtype([('sort','int'), ('id', 'int'), ('per', 'float'), ('TF', 'int')])
xyc_np = np.array(xyc_list, dtype = xyc_dtype)

threshold_list=[0,0,0,0,0,0]

xyc_sort_list=[]

xyc_sort_list.append(xyc_np[827*0:827*1])
xyc_sort_list.append(xyc_np[827*1:827*2])
xyc_sort_list.append(xyc_np[827*2:827*3])
xyc_sort_list.append(xyc_np[827*3:827*4])
xyc_sort_list.append(xyc_np[827*4:827*5])
xyc_sort_list.append(xyc_np[827*5:827*6])

TN=0
TP=0
FN=0
FP=0

# for sort_N in range(6):
#     for threshold in np.arange(0,1,0.001):
#         for row in xyc_sort_list:





pass
