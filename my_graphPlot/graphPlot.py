import os

import pandas as pd
#
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_pandas_Function import *
from myFunctionFolder.my_OS_Function import *
#
predictOutput_Path="predictOutput.npy"
np_predictOutput=readFromNpyFile(predictOutput_Path)

# list_sixPredict=pd.DataFrame( np_predictOutput)
# print(list_sixPredict)

#置入到列表中，列表中每个都是pd。之后对pd排序
listPd_sixPredict=[]
for col in np_predictOutput.T:
    col=pd.DataFrame(col,columns=["probability"])
    col=col.sort_values(by="probability",ascending=False)
    col["TN"]=pd.Series()
    col["TP"]=pd.Series()
    col["FN"]=pd.Series()
    col["TP"]=pd.Series()
    listPd_sixPredict.append(col)

# #输出到csv
# for idx,Pd in enumerate(listPd_sixPredict):
#     outputFile=r".\output"
#     filePath=pathDownToByList(outputFile,["pre_{}.csv".format(idx)])
#     pdWriteToCsv(Pd,filePath,isOpen=0)
#
# osOpen(".\output")
# pass

