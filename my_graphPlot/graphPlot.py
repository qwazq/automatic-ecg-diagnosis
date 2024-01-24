import os

import numpy as np
import pandas as pd
#
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_pandas_Function import *
from myFunctionFolder.my_OS_Function import *

#
predictOutput_Path = "predictOutput.npy"
np_predict = readFromNpyFile(predictOutput_Path)
np_predict=np_predict.T

gold_standard_Path="gold_standard.csv"
np_gold=csvToNp("gold_standard.csv")
np_gold=np_gold.T

pass
# 置入到列表中，列表中每个都是pd。之后对pd排序
listPd_sixPredict = []

for count in range(len(np_gold)):
    pdDf = pd.DataFrame()

    pdDf["probability"]=pd.Series(np_predict[count])
    pdDf["gold"]=pd.Series(np_gold[count])

    #按照probability从大到小
    pdDf = pdDf.sort_values(by="probability", ascending=False)

    pdDf["TN"] = pd.Series()
    pdDf["TP"] = pd.Series()
    pdDf["FN"] = pd.Series()
    pdDf["FP"] = pd.Series()
    listPd_sixPredict.append(pdDf)

pass
# #输出到csv
# for idx,Pd in enumerate(listPd_sixPredict):
#     outputFile=r".\output"
#     filePath=pathDownToByList(outputFile,["pre_{}.csv".format(idx)])
#     pdWriteToCsv(Pd,filePath,isOpen=0)
#
# osOpen(".\output")
# pass

for N_pd, pd in enumerate(listPd_sixPredict):  # 六疾病遍历
    n_all = pd.shape[0]
    n_all_T = pd["gold"].sum()
    n_all_F = n_all-n_all_T
    pass
    for index, row in pd.iterrows():  # 按行遍历
        print("{}:{},{},{},{},{},{}".format(index, row['probability'],row['gold'], row['TN'], row['TP'], row['FN'], row['FP']))
        # break
    pass

    break
pass
