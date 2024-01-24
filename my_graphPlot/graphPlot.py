import os

import numpy as np
import pandas as pd
#
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_pandas_Function import *
from myFunctionFolder.my_OS_Function import *

#
##输入的预测文件，转为np.T
predictOutput_Path = "predictOutput.npy"
np_predict = readFromNpyFile(predictOutput_Path)
np_predict=np_predict.T

##输入的金标准文件，转为np.T
gold_standard_Path="gold_standard.csv"
np_gold=csvToNp("gold_standard.csv")
np_gold=np_gold.astype("bool_")
np_gold=np_gold.T

##输出的六疾病整合csv
outputFolder=".\output"
fileName_sixInfo="sixInfo.csv"
filePath_sixInfo=pathDownToByList(outputFolder,[fileName_sixInfo])
csvW_sixInfo=getCsvWriter(filePath_sixInfo)
csvW_sixInfo.writerow(["病","n_all_T","n_all_F"])

pass
# 六疾病的预测和金标准置入到列表中，列表中每个元素都是pd.df。之后对pd排序
listPd_sixPredict = []

for count in range(len(np_gold)):
    pdDf = pd.DataFrame()

    pdDf["probability"]=pd.Series(np_predict[count])
    pdDf["gold"]=pd.Series(np_gold[count])

    #按照probability从大到小
    pdDf = pdDf.sort_values(by="probability", ascending=False)

    #
    pdDf["TP"] = pd.Series()
    pdDf["TN"] = pd.Series()
    pdDf["FP"] = pd.Series()
    pdDf["FN"] = pd.Series()
    listPd_sixPredict.append(pdDf)

pass


# 六疾病遍历计算个参数
for N_pd, pd in enumerate(listPd_sixPredict):
    n_all = pd.shape[0]
    n_all_T = pd["gold"].sum()
    n_all_F = n_all-n_all_T

    csvW_sixInfo.writerow([N_pd,n_all_T,n_all_F])

    # 按行遍历
    n_cout=0
    for index, row in pd.iterrows():
        # print(row.index)
        pass
        # print("{},{}: {},{},{},{},{},{}".format(n_cout,index, row['probability'],row['gold'], row['TN'], row['TP'], row['FN'], row['FP']))

        pd['TP'][n_cout]=pd["gold"][0:n_cout].sum()
        # row['TN']
        # print(pd['TN'][index])
        n_cout+=1
        break
    pass
    # pd['TN']=n_all_T-pd["TP"]
    break
pass

# 输出到csv
for idx,Pd in enumerate(listPd_sixPredict):
    filePath=pathDownToByList(outputFolder,["pre_{}.csv".format(idx)])
    pdWriteToCsv(Pd,filePath,isOpen=0)
#
# osOpen(outputFolder)
# pass

pass
