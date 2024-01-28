import os

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
#
import warnings

# warnings.filterwarnings("ignore")
#
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_pandas_Function import *
from myFunctionFolder.my_OS_Function import *
from myFunctionFolder.my_Console_Function import *

#
##输入的预测文件，转为np.T
predictOutput_Path = r"..\1 预测\predictOutput_3.npy"
np_predict = readFromNpyFile(predictOutput_Path)
np_predict = np_predict.T

##输入的金标准文件，转为np.T
gold_standard_Path = "./gold_standard.csv"
np_gold = csvToNp(gold_standard_Path)
np_gold = np_gold.astype("bool_")
np_gold = np_gold.T

##输出的六疾病整合csv
outputFolder = ".\六类指标_3"
fileName_sixInfo = "sixInfo.csv"
filePath_sixInfo = pathDownToByList(outputFolder, [fileName_sixInfo])
csvW_sixInfo = getCsvWriter(filePath_sixInfo)
csvW_sixInfo.writerow(["ill", "n_all_T", "n_all_F"])

pass
# 六疾病的预测和金标准置入到列表中，列表中每个元素都是pd.df。之后对pd排序
listPd_sixPredict = []

for count in range(len(np_gold)):
    pdDf = pd.DataFrame()

    pdDf["personId"] = pd.Series(range(len(np_gold.T)), dtype="int")
    pdDf["probability"] = pd.Series(np_predict[count], dtype="float")
    pdDf["gold"] = pd.Series(np_gold[count], dtype="bool")

    # 按照probability从大到小，并重置索引
    pdDf = pdDf.sort_values(by="probability", ascending=False)
    pdDf = pdDf.reset_index(drop=True)

    #
    pdDf["TP"] = pd.Series(dtype="int")
    pdDf["FP"] = pd.Series(dtype="int")
    pdDf["TN"] = pd.Series(dtype="int")
    pdDf["FN"] = pd.Series(dtype="int")

    pdDf["Accuracy"] = pd.Series(dtype="float")
    pdDf["Precision"] = pd.Series(dtype="float")
    pdDf["recall"] = pd.Series(dtype="float")

    pdDf["TN_recall"] = pd.Series(dtype="float")
    pdDf["f1"] = pd.Series(dtype="float")

    #
    listPd_sixPredict.append(pdDf)

pass



# 六疾病遍历计算个参数
for N_pd, pd in enumerate(listPd_sixPredict):
    n_all = pd.shape[0]
    n_all_T = pd["gold"].sum()
    n_all_F = n_all - n_all_T

    csvW_sixInfo.writerow([N_pd, n_all_T, n_all_F])

    re=reporter_V3(allCount=len(pd),Input={"nowSort":N_pd})
    # 按行遍历
    for index, row in pd.iterrows():
        pd['TP'][index] =int(pd["gold"][0:index + 1].sum())
        pd['FP'][index] = int(index + 1 - pd['TP'][index])

        re.AddShow()
    pass
    pd['TN'] =n_all_T - pd["TP"]
    pd['FN'] =n_all_F - pd["FP"]
    #
    pd["Accuracy"] =(pd["TP"]+pd["TN"])/n_all
    pd["Precision"] =pd["TP"]/(pd["TP"]+pd['FP'])
    pd["recall"] =pd["TP"]/(pd["TP"]+pd['FN'])
    pd["TN_recall"] =pd["FN"] / (n_all_F)
    pd["f1"] =2*pd["TP"]/(2*pd["TP"]+pd['FN']+pd['FP'])

pass

# 输出到csv
for idx, Pd in enumerate(listPd_sixPredict):
    filePath = pathDownToByList(outputFolder, ["pre_{}.csv".format(idx)])
    pdWriteToCsv(Pd, filePath, isOpen=0)

osOpen(outputFolder)
pass

# osOpen(
#     pathDownToByList(outputFolder, ["pre_0.csv"]),
#     # software="excel"
# )


print("over")
pass

