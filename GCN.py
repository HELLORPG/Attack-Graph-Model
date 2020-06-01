"""
本文件用于训练一个GCN，用于分类节点。
完成分类任务的训练之后，使用这个分类器去完成后续的attack。
"""

import torch
import OpData
from torch_geometric.data import Data   # 用于承载Graph Data


def GraphData() -> Data:
    """
    :return: 得到一个torch_geometric框架下支持的Graph Data
    """
    X = torch.tensor(OpData.GetFeatures(), dtype=torch.float)
    Y = torch.tensor(OpData.GetLabels(), dtype=torch.int)
    edges = torch.tensor(OpData.GetAdj(), dtype=torch.long)
    data = Data(x=X, y=Y, edge_index=edges)
    data.to(torch.device('cuda'))
    return data


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  # 调用父类初始化


def main():
    data = GraphData()


main()
