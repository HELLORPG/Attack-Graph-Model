import GCN
import torch
import torch_geometric

from GCN import GCNClassifier   # 一定需要引入这个class才能load model

from CONFIG import CONFIG
from torch_geometric.data import Data   # 用于承载Graph Data
from torch_geometric.nn import GraphConv, MessagePassing, GCNConv
import torch.nn.functional as F


class Attacker(torch.nn.Module):
    def __init__(self):
        super(GCNClassifier, self).__init__()  # 调用父类初始化
        self.conv1 = GCNConv(CONFIG.FeatureLen(), 180)
        self.conv2 = GCNConv(180, 120)
        self.lin1 = torch.nn.Linear(120, CONFIG.ClassNum())

    def forward(self, data: Data):
        """
        :param data: 传入数据集
        :return:
        """
        x = data.x
        x = self.conv1(x, data.edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.leaky_relu(x)
        x = self.lin1(x)
        x = F.log_softmax(x, dim=1)
        torch.cuda.empty_cache()
        return x



if __name__ == '__main__':
    # model = GCN.GCNClassifier()
    gcn_model = torch.load("Models/model-conv-conv-line-0.45")
    print(">>>>> Load Model Finish")
    attack_model = Attacker()
    attack_dict = attack_model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)