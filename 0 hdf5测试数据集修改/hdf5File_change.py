#
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_OS_Function import *
from myFunctionFolder.my_hdf5_Function import *

#
folderPath_testDateSet=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data"
fileName_testDateSet=r"ecg_tracings.hdf5"
path_testDateSet=joinPathAndName(folderPath_testDateSet,fileName_testDateSet)
# path_testDateSet= r"ecg_tracings_6.hdf5"  #
#
groupName_testDateSet="tracings"

print(path_testDateSet)
np_testDateSet=np_from_hdf5(path_testDateSet, groupName_testDateSet)
#
#hdf5模型中的导联数
n_link=6;

np_testDateSet[:,:,n_link:]=0

#标准化
def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)

    return (data - mu) / sigma

# #归一化
# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range

np_testDateSet=standardization(np_testDateSet)
#展示
np2DToCsv(np_testDateSet[0],"test_正则化后.csv",software="excel")
# np_writeTo_hdf5(np_testDateSet,r"ecg_tracings_{}.hdf5".format(n_link),groupName_testDateSet)
np_writeTo_hdf5(np_testDateSet,r"ecg_tracings_{}{}.hdf5".format(n_link,"_stand"),groupName_testDateSet)

#
print(np_testDateSet.shape)