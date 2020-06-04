import pickle   # 用于读取pkl文件
import numpy as np
import scipy.sparse.csr
import random


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


def ReadAdj() -> scipy.sparse.csr.csr_matrix:
    adj = ReadPkl("Data/experimental_adj.pkl")
    return adj


def TestOutput():
    # 543486 ~ 593486 is the target
    x = []
    y = []
    data = []
    for i in range(0, 500):
        for j in range(0, 100):
            # dst = random.randint(543486, 593485)
            dst = 543486 + i * 100 + j
            x.append(i)
            y.append(dst)
            # print(x, y)
            # x.append(dst)
            # y.append(i)
            # data.append(1)
            data.append(1)
    adj = scipy.sparse.coo_matrix((data, (x, y)), shape=(500, 593986))
    adj = adj.tocsr()
    # print(np.transpose(adj[:, 593486:]) == adj[:, 593486:])
    with open("adj.pkl", 'wb') as f:  # 将数据写入pkl文件
        pickle.dump(adj, f)
    features = np.zeros((500, 100))
    for i in range(0, 500):
        for j in range(0, 100):
            flag = random.randint(0, 1)
            if flag == 0:
                features[i][j] = 99.9
            else:
                features[i][j] = -99.9
    print(features)
    np.save("feature.npy", features)


# ReadPkl("Data/experimental_adj.pkl")
# ReadPkl("Data/experimental_features.pkl")
# ReadPkl("Data/experimental_train.pkl")


if __name__ == '__main__':
    TestOutput()