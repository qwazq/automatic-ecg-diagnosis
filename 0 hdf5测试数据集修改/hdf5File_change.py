#
from myFunctionFolder.my_CSV_Function import *
from myFunctionFolder.my_OS_Function import *
from myFunctionFolder.my_hdf5_Function import *

#
#folderPath_testDateSet=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data"
#fileName_testDateSet=r"ecg_tracings.hdf5"
path_testDateSet=r"./ecg_tracings_only6Line.hdf5"#joinPathAndName(folderPath_testDateSet,fileName_testDateSet)

#

groupName_testDateSet="tracings"

print(path_testDateSet)
np_testDateSet=np_from_hdf5(path_testDateSet, groupName_testDateSet)
#

# np_testDateSet[:,:,6:12]=0

np2DToCsv(np_testDateSet[0],"test.csv",software="excel")
# np_writeTo_hdf5(np_testDateSet,r"ecg_tracings_only6Line.hdf5",groupName_testDateSet)