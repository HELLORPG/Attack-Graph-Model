"""
本文件用于训练一个GCN，用于分类节点。
完成分类任务的训练之后，使用这个分类器去完成后续的attack。
"""

import torch
import OpData
import torch_geometric

# import torch_xla
# import torch_xla.core.xla_model as xm

from CONFIG import CONFIG
from torch_geometric.data import Data   # 用于承载Graph Data
from torch_geometric.nn import GraphConv, MessagePassing, GCNConv
import torch.nn.functional as F


# device = xm.xla_device()

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
    # data.to(torch.device('cuda'))
    data.to(torch.device('cpu'))
    return data


class GCNClassifier(torch.nn.Module):
# class Classifier(MessagePassing):
    def __init__(self):
        super(GCNClassifier, self).__init__()  # 调用父类初始化
        self.conv1 = GCNConv(CONFIG.FeatureLen(), 180)
        self.conv2 = GCNConv(180, 120)
        # self.conv3 = GCNConv(100, CONFIG.ClassNum())
        self.lin1 = torch.nn.Linear(120, CONFIG.ClassNum())
        # self.lin2 = torch.nn.Linear(50,  CONFIG.ClassNum())
        # self.lin3 = torch.nn.Linear(40, CONFIG.ClassNum())
        # self.lin4 = torch.nn.Linear(50, 50)
        # self.lin5 = torch.nn.Linear(50, CONFIG.ClassNum())

    def forward(self, data: Data):
        """
        :param data: 传入数据集
        :return:
        """
        # x, edges = data.x, data.edge_index
        x = data.x

        x = self.conv1(x, data.edge_index)
        x = F.leaky_relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)

        # # print("shape of x:", x.shape)
        # # print("shape of y:", data.y.shape[0])

        x = F.leaky_relu(x)
        # # x = F.dropout(x, training=self.training)

        # x = self.conv3(x, edges)
        # # x = F.relu(x)
        x = self.lin1(x)
        # x = F.leaky_relu(x)
        # x = self.lin2(x)
        # x = F.leaky_relu(x)
        # x = self.lin3(x)

        # x = F.leaky_relu(x)
        # x = self.lin4(x)

        # x = F.leaky_relu(x)
        # x = self.lin5(x)

        x = F.log_softmax(x, dim=1)
        # print(x.shape)
        torch.cuda.empty_cache()
        return x


if __name__ == '__main__':
    data = GraphData()
    # device = torch.device('cuda')
    device = torch.device('cpu')
    model = GCNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    model.train()
    for i in range(0, 2000):
        optimizer.zero_grad()
        out = model(data)[:data.y.shape[0] - 50000, :]
        loss = F.nll_loss(out, data.y.squeeze()[:data.y.shape[0] - 50000])
        # print(loss.data)
        loss.backward()
        optimizer.step()
        if i%10 == 9:
          _, pred = model(data).max(dim=1)
          # print(pred.shape)
          pred_on_test = pred[data.y.shape[0] - 50000: data.y.shape[0]]
          pred = pred[:data.y.shape[0] - 50000]
          
          correct = float(pred.eq(data.y.squeeze()[:data.y.shape[0]-50000]).sum().item())
          acc = correct / (data.y.shape[0] - 50000)
          correct_on_test = float(pred_on_test.eq(data.y.squeeze()[data.y.shape[0]-50000:data.y.shape[0]]).sum().item())
          # acc_on_test = float(pred_on_test.eq(data.y.squeeze()[data.y.shape[0] - 50000:data.y.shape[0]]).sum().item()) / 50000
          acc_on_test = correct_on_test / 50000
          print("Train Turn: %d" % (i+1), "Accuracy: {:.6f}".format(acc), "Accuracy on test: {:.6f}".format(acc_on_test))
          del _, pred, correct, acc
        del loss, out
        torch.cuda.empty_cache()
    model.eval()

    torch.save(model, "model")





