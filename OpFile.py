import pickle   # 用于读取pkl文件
import numpy as np


def ReadPkl(filename: str):
    file = open(filename, "rb")
    data = pickle.load(file)
    # 对于邻接矩阵文件，这里可以读取到邻接矩阵形式，例如：(4, 192082)	1
    # 对于特征文件，这里可以读取到ndarray形式，二维
    # 对于标签文件，这里可以读取到ndarray形式，一维
    # print(type(data))
    # print(data)
    return data


def ReadLabels() -> np.ndarray:
    """
    :return: 读入真实标签
    """
    return ReadPkl("Data/experimental_train.pkl")


def ReadFeatures() -> np.ndarray:
    """
    :return: 返回提供的数据集的Features
    """
    return ReadPkl("Data/experimental_features.pkl")


# ReadPkl("Data/experimental_adj.pkl")
# ReadPkl("Data/experimental_features.pkl")
# ReadPkl("Data/experimental_train.pkl")


