import torch
import GCN
import torch_geometric
import OpData
import ClusteringTarget
import scipy.sparse.csr
import pickle

import numpy as np

from GCN import GCNClassifier   # 一定需要引入这个class才能load model

from CONFIG import CONFIG
from torch_geometric.data import Data   # 用于承载Graph Data
from torch_geometric.nn import GraphConv, MessagePassing, GCNConv
import torch.nn.functional as F

device = torch.device('cpu')


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
    data.to(device)
    return data


class Attacker(torch.nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()  # 调用父类初始化
        self.attack = False
        self.conv1 = GCNConv(CONFIG.FeatureLen(), 180)
        self.conv2 = GCNConv(180, 120)
        self.lin1 = torch.nn.Linear(120, CONFIG.ClassNum())
        # 这次反而需要调整的参数是输入
        self.X = torch.tensor([1], dtype=torch.float)
        self.edges = torch.tensor([[], []], dtype=torch.long)

    def loadX(self, X: torch.tensor):
        self.X = X
        if self.X.requires_grad is False:
            self.X.requires_grad = True
        return

    def loadEdges(self, edges: torch.tensor):
        self.edges = edges
        self.attack = True
        return

    def forward(self, data: Data):
        """
        :param data: 传入数据集
        :return:
        """
        if self.attack is False:
            edges = data.edge_index
        else:
            edges = torch.cat((data.edge_index, self.edges), 1)     # 横向叠加
        edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)
        # print(edges)
        if self.attack is False:
            x = data.x
        else:
            x = torch.cat((data.x, self.X), 0)  # 0代表纵向叠加
        x = self.conv1(x, edges)
        x = F.leaky_relu(x)
        x = self.conv2(x, edges)
        x = F.leaky_relu(x)
        x = self.lin1(x)
        x = F.log_softmax(x, dim=1)
        return x

    def freeze(self):
        for parameter in self.conv1.parameters():
            parameter.requires_grad = False
        for parameter in self.conv2.parameters():
            parameter.requires_grad = False
        for parameter in self.lin1.parameters():
            parameter.requires_grad = False

    def get_x_grad(self) -> torch.tensor:
        return self.X.grad


def load_add_features():
    data = np.load("feature.npy")
    data = data.astype(np.float32)
    return data


def init_add_edges() -> torch.tensor:
    """
    这是一个最基础的初始化方式，1节点连接1~100测试节点，2连接201~400测试节点，直到50000为止
    :return: 用这个函数来初始化加入的邻接矩阵
    """
    x = []
    y = []
    data = []
    # 前面三个初始化，用于记录起始点x，终点y和权重data
    for i in range(0, 500):
        for j in range(0, 100):
            src = CONFIG.TargetEnd() + i
            dst = CONFIG.TargetBegin() + i * 100 + j
            assert (src >= CONFIG.TargetEnd()) and (src < CONFIG.TargetEnd() + 500)
            assert (dst >= CONFIG.TargetBegin()) and (dst < CONFIG.TargetEnd())
            x.append(src)
            y.append(dst)
            x.append(dst)
            y.append(src)
            data.append(1)
            data.append(1)
    assert len(x) == len(y) == len(data)
    # 将其化为tensor
    edges = []
    edges.append(x)
    edges.append(y)
    return torch.tensor(edges, dtype=torch.long).to(device)


def add_edges_by_cluster():
    cut = ClusteringTarget.cut_cluster_by_100()
    x = []
    y = []

    coo_x = []
    coo_y = []
    data = []

    for i in range(0, 500):
        for j in range(0, 100):
            src = CONFIG.TargetEnd() + i
            dst = CONFIG.TargetBegin() + cut[i][j]
            assert (src >= CONFIG.TargetEnd()) and (src < CONFIG.TargetEnd() + 500)
            assert (dst >= CONFIG.TargetBegin()) and (dst < CONFIG.TargetEnd())
            x.append(src)
            y.append(dst)
            x.append(dst)
            y.append(src)

            coo_x.append(src - CONFIG.TargetEnd())
            coo_y.append(dst)
            data.append(1)

    # 保存邻接矩阵形式：
    adj = scipy.sparse.coo_matrix((data, (coo_x, coo_y)), shape=(500, 593986))
    adj = adj.tocsr()
    with open("adj.pkl", 'wb') as f:  # 将数据写入pkl文件
        pickle.dump(adj, f)

    # 将其化为tensor
    edges = []
    edges.append(x)
    edges.append(y)
    return torch.tensor(edges, dtype=torch.long).to(device)


def add_edges_by_class(labels: torch.tensor) -> torch.tensor:
    labels_cpu = labels.cpu()
    labels_list = labels_cpu.numpy().tolist()
    cut = [[]]

    # class_divide用于将索引划入不同的类别
    class_divide = []
    for i in range(0, CONFIG.ClassNum()):
        class_divide.append([])

    for i in range(0, len(labels_list)):
        class_divide[labels_list[i]].append(i)

    remain_data = []
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
    # 此时，cut里面就是分割好的对照关系

    x = []
    y = []

    coo_x = []
    coo_y = []
    data = []

    for i in range(0, 500):
        for j in range(0, 100):
            src = CONFIG.TargetEnd() + i
            dst = CONFIG.TargetBegin() + cut[i][j]
            assert (src >= CONFIG.TargetEnd()) and (src < CONFIG.TargetEnd() + 500)
            assert (dst >= CONFIG.TargetBegin()) and (dst < CONFIG.TargetEnd())
            x.append(src)
            y.append(dst)
            x.append(dst)
            y.append(src)

            coo_x.append(src - CONFIG.TargetEnd())
            coo_y.append(dst)
            data.append(1)

    # for i in range(CONFIG.TargetBegin(), CONFIG.TargetEnd()):
    #     if i in coo_y:
    #         continue
    #     else:
    #         print("ERROR!!!", i)
    # print("OK!")

    # 保存邻接矩阵形式：
    adj = scipy.sparse.coo_matrix((data, (coo_x, coo_y)), shape=(500, 593986))
    adj = adj.tocsr()
    # print(adj)
    with open("adj.pkl", 'wb') as f:  # 将数据写入pkl文件
        pickle.dump(adj, f)

    # 将其化为tensor
    edges = []
    edges.append(x)
    edges.append(y)
    # print(edges)
    return torch.tensor(edges, dtype=torch.long).to(device)


def fix_features(features: torch.tensor) -> torch.tensor:
    features = features.cpu()
    features = features.numpy()
    for i in range(0, features.shape[0]):
        for j in range(0, features.shape[1]):
            if features[i][j] <= -100.0:
                features[i][j] = -99.9999
            if features[i][j] >= 100.0:
                features[i][j] = 99.9999
    features = torch.from_numpy(features)
    features = features.to(device)
    return features


if __name__ == '__main__':
    data = GraphData()
    # model = GCNClassifier().to(device)
    # model = GCN.GCNClassifier()
    gcn_model = torch.load("Models/model-conv-conv-line-0.45")
    attack_model = Attacker().to(device)
    attack_model.load_state_dict(gcn_model.state_dict())
    # attack_dict = attack_model.state_dict()
    # state_dict = {k: v for k, v in gcn_model.items() if k in attack_model.keys()}
    # print(state_dict.keys())
    # attack_dict.update(state_dict)
    # attack_model.load_state_dict(attack_dict)
    print(">>>>> Load Model Finish")

    # 冻结Attack模型中的参数
    attack_model.freeze()
    print(">>>>> Freeze all parameters in Attacker")

    # 这里得到所有的target的label
    _, pred = attack_model(data).max(dim=1)
    target_labels = pred[CONFIG.TargetBegin(): CONFIG.TargetEnd()]
    print(target_labels)

    # 将需要添加的features载入模型
    add_features = torch.zeros((500, 100), requires_grad=False).to(device)
    attack_model.loadX(add_features)
    # 将需要添加的邻接矩阵edges载入模型
    # add_edges = init_add_edges()
    # add_edges = add_edges_by_cluster()
    add_edges = add_edges_by_class(target_labels)
    attack_model.loadEdges(add_edges)

    # 下面是attack的过程
    while True:
        attack_model.zero_grad()
        attack_model.loadX(add_features)
        model_out = attack_model(data)[CONFIG.TargetBegin(): CONFIG.TargetEnd()]
        loss = F.nll_loss(model_out, target_labels)
        print("Loss: {:.8f}".format(loss.data.tolist()))
        loss.backward()
        x_grad = attack_model.get_x_grad()
        # print("x grad is", x_grad)
        with torch.no_grad():
            add_features = add_features + CONFIG.AttackRate() * x_grad
            add_features = fix_features(add_features)
            print("features change a turn")
            output = add_features.cpu()
            np.save("feature.npy", output.numpy())
