"""
用于进行一些内存内的数据操作。
注意，该文件内的内容应该不涉及内存与磁盘之间的操作：这部分操作应该交给OpFile。
"""

import OpFile
import numpy as np


def GetLabels() -> np.matrix:
    """
    :return: 返回labels，采用np.matrix格式统一
    """
    labels_ndarray = OpFile.ReadLabels()

