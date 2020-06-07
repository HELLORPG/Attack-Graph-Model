"""
该文件内是一些，不需要是使用Models的入侵方式。
俗称，naive method
"""

import numpy as np


def FeaturesRangeAttack():
    """
    该方法企图利用模型的归一化，来进行攻击。
    :return: None
    """
    features = np.zeros((500, 100), dtype=np.float32)
    for i in range(0, 500):
        j = int(i / 5)
        features[i][j] = (-1)**i * 99.99
    print(features)
    np.save("feature_range_attack.npy", features)


if __name__ == '__main__':
    FeaturesRangeAttack()