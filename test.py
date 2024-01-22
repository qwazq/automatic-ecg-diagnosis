#导入所需的包


from myFunctionFolder.my_numpy_Function import *
from myFunctionFolder.my_OS_Function import *

#导入npy文件路径位置
test = np.load(r"my_graphPlot/predictOutput.npy")
print(test.shape)

np2DToCsv(r"./predictOutput.npy", r"./predictOutput.csv")


osOpen(r"./predictOutput.csv")
# print(test)