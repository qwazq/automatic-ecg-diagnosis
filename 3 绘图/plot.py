import os

import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
#
import warnings
# warnings.filterwarnings("ignore")
#
from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_pandas_Function import *
from myFunctionFolder.my_OS_Function import *

#
##输入的预测文件，转为np.T
path_output = r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_code\2 predict计算指标\六类指标_3"

pass
# 六疾病的预测和金标准置入到列表中，列表中每个元素都是pd.df。之后对pd排序
listPd_sixPredict = []
print(pathJoin(path_output, "pre_", "0", ".csv"))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "0", ".csv"),engine='python'))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "1", ".csv"),engine='python'))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "2", ".csv"),engine='python'))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "3", ".csv"),engine='python'))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "4", ".csv"),engine='python'))
listPd_sixPredict.append(pd.read_csv(pathJoin(path_output, "pre_", "5", ".csv"),engine='python'))

# print(listPd_sixPredict[0])

plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ['SimHei']

for N_pddf, pddf in enumerate(listPd_sixPredict):
    plt.clf()
    plt.cla()

    plt.plot(pddf["probability"], pddf["TN_recall"], label="TN_recall")#=FN/(FN+FP))
    plt.plot(pddf["probability"], pddf["Precision"], label="Precise")#=TP/(TP+FP)
    plt.plot(pddf["probability"], pddf["f1"], label="f1")

    plt.xlim((-0.1, 1.1))
    # plt.ylim((-0.1, 1.1))
    plt.xlabel('阈值threshold')
    plt.ylabel('')

    # plt.xticks(np.arange(0, 1, 0.001))
    # plt.yticks(np.arange(0, 1, 0.001))

    plt.legend(
        # bbox_to_anchor=(1.02, 0.1),
        loc='upper right',
        borderaxespad=0
               )

    str_title = "瑞典827人的数据集,3导联,第{}种类型".format(N_pddf)
    plt.title(str_title)

    plt.savefig(pathJoin(path_output, str_title))
    # break

osOpen(path_output)
pass
