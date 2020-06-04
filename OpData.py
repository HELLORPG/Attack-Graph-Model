"""
用于进行一些内存内的数据操作。
注意，该文件内的内容应该不涉及内存与磁盘之间的操作：这部分操作应该交给OpFile。
"""

import OpFile
import numpy as np
from CONFIG import CONFIG


def GetLabels() -> np.matrix:
    """
    :return: 返回labels，采用np.matrix格式统一
    """
    labels_ndarray = OpFile.ReadLabels()
    labels_matrix = labels_ndarray.reshape((len(labels_ndarray), 1))
    return labels_matrix


def GetFeatures() -> np.matrix:
    features_ndarray = OpFile.ReadFeatures()
    features_matrix = features_ndarray.reshape((len(features_ndarray), CONFIG.FeatureLen()))
    return features_matrix


def GetAdj() -> np.matrix:
    adj_csr = OpFile.ReadAdj()
    adj_coo = adj_csr.tocoo()
    links_num = adj_coo.getnnz()
    print(links_num)
    # print(adj_coo.getnnz())     # 通过这行语句可以查看非零元素的个数
    # 这里整个数据如果采用密集矩阵的形式，约有2.56TB
    src = adj_coo.row
    dst = adj_coo.col
    links = list()
    links.append(src.tolist())
    links.append(dst.tolist())
    edges = np.mat(links)
    assert edges.shape == (2, links_num)
    return edges


if __name__ == '__main__':
    # print(GetLabels())
    # print(GetFeatures())
    # GetAdj()
    # print("Min Class Label: %d, Max Class Label: %d" % (GetLabels().min(), GetLabels().max()))
    # print("OpData Test Finish")
    y = GetFeatures()
    yty = np.dot(y.T, y)
    print("yty.shape =", yty.shape)
    lamda, w = np.linalg.eig(yty)
    lamda.sort()
    print(lamda)
    GetAdj()
