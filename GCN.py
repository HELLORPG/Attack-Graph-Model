"""
本文件用于训练一个GCN，用于分类节点。
完成分类任务的训练之后，使用这个分类器去完成后续的attack。
"""

import torch
import OpFile


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  # 调用父类初始化

