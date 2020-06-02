"""
本文件用于训练一个GCN，用于分类节点。
完成分类任务的训练之后，使用这个分类器去完成后续的attack。
"""

import torch
import OpData
import torch_geometric

from CONFIG import CONFIG
from torch_geometric.data import Data   # 用于承载Graph Data
from torch_geometric.nn import GraphConv, MessagePassing, GCNConv
import torch.nn.functional as F


def GraphData() -> Data:
    """
    :return: 得到一个torch_geometric框架下支持的Graph Data
    """
    X = torch.tensor(OpData.GetFeatures(), dtype=torch.float)
    Y = torch.tensor(OpData.GetLabels(), dtype=torch.long)
    edges = torch.tensor(OpData.GetAdj(), dtype=torch.long)     # 这两个Long是不可以更改的
    # print(edges.shape)
    edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)    # 该操作只增加当前不存在的自环
    # print(edges.shape)
    # edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)
    # print(edges.shape)
    data = Data(x=X, y=Y, edge_index=edges)
    data.to(torch.device('cuda'))
    return data


class GCNClassifier(torch.nn.Module):
# class Classifier(MessagePassing):
    def __init__(self):
        super(GCNClassifier, self).__init__()  # 调用父类初始化
        self.conv1 = GCNConv(CONFIG.FeatureLen(), 80)
        self.conv2 = GCNConv(80, 40)
        self.conv3 = GCNConv(40, 50)
        self.lin1 = torch.nn.Linear(50, CONFIG.ClassNum())
        # self.lin2 = torch.nn.Linear(10, 20)
        # self.lin3 = torch.nn.Linear(20, CONFIG.ClassNum())

    def forward(self, data: Data):
        """
        :param data: 传入数据集
        :return:
        """
        x, edges = data.x, data.edge_index

        x = self.conv1(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edges)

        # print("shape of x:", x.shape)
        # print("shape of y:", data.y.shape[0])

        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edges)
        x = F.leaky_relu(x)
        x = self.lin1(x)
        # x = F.leaky_relu(x)
        # x = self.lin2(x)
        # x = F.leaky_relu(x)
        # x = self.lin3(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    data = GraphData()
    device = torch.device('cuda')
    model = GCNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for i in range(0, 5000):
        optimizer.zero_grad()
        out = model(data)[:data.y.shape[0], :]
        loss = F.nll_loss(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        _, pred = model(data).max(dim=1)
        pred = pred[:data.y.shape[0]]
        correct = float(pred.eq(data.y.squeeze()).sum().item())
        acc = correct / data.y.shape[0]
        print("Train Turn: %d" % i, "Accuracy: {:.4f}".format(acc))
    model.eval()





