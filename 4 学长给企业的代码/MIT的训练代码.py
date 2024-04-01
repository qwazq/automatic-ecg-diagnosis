import os
import time
import wfdb
import pywt
import seaborn
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.signal
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 测试集在数据集中所占的比例
RATIO = 0.2
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)


def denoise(data):
    """
    小波去噪预处理
    :param data: 一维时序数据
    :return: 降噪后的数据
    """
    # 进行9层的小波分解
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def getNoiseSet(number, X_data, Y_data, flag):
    """
    读取噪音数据（不是根据R波切割的方式），并对数据进行小波降噪
    :param number: 读取第几号噪音数据
    :param X_data: 样本数据集
    :param Y_data: 标签数据集
    :param flag: 1和2代表不同的噪音数据 噪音的路径需要修改一下
    """
    print("正在读取 " + str(number) + " 号噪音数据...")
    if flag == 1:
        data = np.load('/root/tots/ecg/examples/noise2/' + str(number) + '_lien.npy')
    elif flag == 2:
        data = np.load('/root/tots/ecg/examples/noise22/lien_' + str(number) + '.npy')
    data = (data - np.mean(data)) / np.std(data)
    n = len(data) // 108
    for i in range(n):
        x_train = data[i*108:(i+1)*108]
        x_train = scipy.signal.resample(x_train, 300)
        x_train = denoise(x_train)
        X_data.append(x_train)
        Y_data.append(5)


# 读取噪音数据, 使用R波降噪，并对数据进行小波降噪
# flag 1和2代表不同的噪音数据 噪音的路径需要修改一下
def getNoiseForRSet(number, X_data, Y_data, flag):
    """
    读取噪音数据，根据R波切割的方式，并对数据进行小波降噪
    :param number: 读取第几号噪音数据
    :param X_data: 样本数据集
    :param Y_data: 标签数据集
    :param flag: 1和2代表不同的噪音数据 噪音的路径需要修改一下
    """
    print("正在读取 " + str(number) + " 号噪音数据...")
    if flag == 1:
        data = np.load('/root/tots/ecg/examples/noise2/' + str(number) + '_lien.npy')
        R = np.load('/root/tots/ecg/examples/noise2/' + str(number) + '_pan.npy')
    elif flag == 2:
        data = np.load('/root/tots/ecg/examples/noise22/lien_' + str(number) + '.npy')
        R = np.load('/root/tots/ecg/examples/noise22/pan_' + str(number) + '.npy')
    data = (data - np.mean(data)) / np.std(data)
    for i in R:
        if 36 < i < len(data)-73:
            x_train = data[int(i)-36:int(i)+72]
            x_train = scipy.signal.resample(x_train, 300)
            x_train = denoise(x_train)
            X_data.append(x_train)
            Y_data.append(5)


def getDataSet(number, X_data, Y_data):
    """
    读取心电数据和对应标签,并对数据进行小波去噪
    :param number: 心电数据记录的编号
    :param X_data: 训练数据集
    :param Y_data: 标签数据集
    :return:
    """

    # 正常节拍/房性早搏/室性早搏/左束支传导阻滞/右束支阻滞
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    # 读取MLII导联的数据
    record = wfdb.rdrecord('/root/tots/ecg/examples/mitdb/data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('/root/tots/ecg/examples/mitdb/data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            # 基于经验值，基于R峰向前取100个点，向后取200个点
            x_train = rdata[Rlocation[i] - 100:Rlocation[i] + 200]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


def loadData():
    """
    加载MIT数据集并进行预处理
    或加载自己处理的数据集
    :return: 样本训练集，标签训练集，样本测试集，标签测试集
    """
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    noiseSet = 32
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 读取噪音数据
    # for n in range(32):
    #     # getNoiseSet(n, dataSet, lableSet, 1)
    #     getNoiseForRSet(n, dataSet, lableSet, 1)
    # for n in range(61):
    #     # getNoiseSet(n, dataSet, lableSet, 2)
    #     getNoiseForRSet(n, dataSet, lableSet, 2)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)
    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


# 构建CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='tanh'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='tanh'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理'
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点 转换成128个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(5, activation='softmax')
        # 带噪音
        # tf.keras.layers.Dense(6, activation='softmax')
    ])
    return newModel


def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 绘图
    plt.figure(figsize=(4, 5))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def get_filename_for_saving(save_dir):
    """
    返回文件夹路径+损失率准确率的字符串作为存储模型的名称
    :param save_dir:
    :return: 文件夹路径+损失率准确率
    """
    return os.path.join(save_dir,
            "R-{val_loss:.3f}-{val_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{accuracy:.3f}.h5")


def make_save_dir(dirname, experiment_name):
    """
    根据时间戳和随机数创建目标文件夹并返回文件夹路径
    :param dirname: 保存的文件夹
    :param experiment_name: 保存的文件类型
    :return: dirname + experiment_name + start_time
    """
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def main():
    """
    主程序运行，包括模型的训练，测试，保存
    """

    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()

    # 构建模型
    model = buildModel()
    # 选择优化方式
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy']
                  # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
                  )
    model.summary()

    # 当八轮没有进步时停止训练
    stopping = tf.keras.callbacks.EarlyStopping(patience=8)

    # 模型保存路径
    save_dir = make_save_dir("saved", "new")
    # 在每个epoch之后保存模型
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    # 训练与验证
    model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_split=RATIO, callbacks=[checkpointer, stopping])  # validation_split 训练集所占比例
    # 预测
    scores = model.evaluate(X_test, Y_test)
    print('%s: %.2f%%' % (model.metrics_names[0], scores[0] * 100))
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    # main()
    save_dir = make_save_dir("saved", "new")
    print(save_dir)