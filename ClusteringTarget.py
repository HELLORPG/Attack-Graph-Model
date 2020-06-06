"""
该文件用于，使用聚类方法对Target中的样本进行i.i.d聚类
"""

import numpy as np
from sklearn.cluster import KMeans
import OpData
from CONFIG import CONFIG


def Cluster():
    # 得到需要聚类的样本的Features
    features = OpData.GetFeatures()
    features = features[CONFIG.TargetBegin(): CONFIG.TargetEnd()]
    # print(features.shape)

    # 构造一个聚类分类器
    cluster = KMeans(n_clusters=CONFIG.ClassNum())
    # cluster.fit(features)
    result = cluster.fit_predict(features)

    # 将聚类结果输出
    np.save("cluster_result.npy", result)
    np.savetxt("cluster_result.csv", result, delimiter=',', fmt="%d")

    class_num = np.zeros((CONFIG.ClassNum()))
    for r in result:
        class_num[r] += 1
    np.savetxt("class_num.csv", class_num, delimiter=',', fmt="%d")


def cut_cluster_by_100() -> list:
    """
    :return: 返回一个500*100的list标记edges的构造
    """
    cut = [[]]
    class_divide = []
    for i in range(0, 18):
        class_divide.append([])
    cluster = np.load("cluster_result.npy")

    # print(cluster)
    cluster = cluster.tolist()
    # print(cluster)
    for c in range(0, len(cluster)):
        class_divide[cluster[c]].append(c)
    # print(class_divide)

    remain_data = []     # 用来记录遗留下的数据，用于二次划分
    cut_i = 0

    for i in range(0, CONFIG.ClassNum()):
        # 用于遍历cluster
        for j in range(0, len(class_divide[i])):
            if j < int(len(class_divide[i]) / 100) * 100:
                cut[cut_i].append(class_divide[i][j])
                if len(cut[cut_i]) is 100:
                    cut_i += 1
                    cut.append([])
            else:
                # print(j)
                remain_data.append(class_divide[i][j])
    for r in remain_data:
        cut[cut_i].append(r)
        # print(len(cut))
        if (len(cut[cut_i]) is 100) and (len(cut) != 500):
            cut_i += 1
            cut.append([])

    # print(cut)
    # for cut_ in cut:
    #     print(len(cut_))

    return cut


if __name__ == '__main__':
    # Cluster()
    cut_cluster_by_100()
