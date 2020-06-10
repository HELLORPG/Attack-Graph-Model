"""
该文件用于构造一个，基于文件GCN.py中的架构，而更规范化的一个分类器。
用作该任务中的基准分类器。
也作为假想的攻击目标。
"""

import torch
import OpData
import torch_geometric
import torch.nn.functional as F

from CONFIG import CONFIG
from torch_geometric.data import Data   # 用于承载Graph Data
from torch_geometric.nn import GCNConv


device = torch.device('cpu')
# 使用这个device变量来统一管理GCN下的设备划分
LABEL_LEN = 543486  # 已知的标签长度
TRAIN_TURNS = 200


def train_valid_test():
    """
    :return: 返回数据的训练集/验证集/测试集划分
    """
    return 0.1, 0.1, 0.8


def valid_begin_test_begin():
    """
    :return: 返回验证集和测试集的起始位置
    """
    train, valid, test = train_valid_test()
    valid_begin = int(train * LABEL_LEN)
    test_begin = int((train + valid) * LABEL_LEN)
    return valid_begin, test_begin


def get_all_data() -> Data:
    X = torch.tensor(OpData.GetFeatures(), dtype=torch.float)
    Y = torch.tensor(OpData.GetLabels(), dtype=torch.long)
    edges = torch.tensor(OpData.GetAdj(), dtype=torch.long)
    edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)  # 该操作只增加当前不存在的自环
    data = Data(x=X, y=Y, edge_index=edges)
    data.to(device)
    return data


class RPGGCN(torch.nn.Module):
    def __init__(self, drop=0.5, conv1_hide=100, leaky=0.01):
        super(RPGGCN, self).__init__()  # 调用父类
        self.drop = drop    # 在dropout操作中，元素被置为0的概率
        self.leaky = leaky
        self.conv1 = GCNConv(CONFIG.FeatureLen(), conv1_hide)
        self.conv2 = GCNConv(conv1_hide, CONFIG.ClassNum())


    def forward(self, data: Data):
        """
        :param data: 传入的数据集
        :return: 运算结果
        """
        x = data.x

        x = self.conv1(x, data.edge_index)
        x = F.leaky_relu(x, negative_slope=self.leaky)
        x = F.dropout(x, training=self.training, p=self.drop)

        x = self.conv2(x, data.edge_index)

        torch.cuda.empty_cache()

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    data = get_all_data()
    model = RPGGCN(drop=0.5, conv1_hide=100, leaky=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

    valid_begin, test_begin = valid_begin_test_begin()
    model.train()
    for i in range(0, TRAIN_TURNS):
        optimizer.zero_grad()   # 清零梯度
        out = model(data)[:test_begin, :]
        train_out = out[:valid_begin, :]
        valid_out = out[valid_begin: test_begin, :]
        del out
        torch.cuda.empty_cache()

        loss = F.nll_loss(train_out, data.y.squeeze()[: valid_begin])
        loss.backward()
        optimizer.step()

        del loss
        torch.cuda.empty_cache()

        if i % 10 is 9:
            _, pred_train = train_out.max(dim=1)
            _, pred_valid = valid_out.max(dim=1)

            train_correct = float(pred_train.eq(data.y.squeeze()[:valid_begin]).sum().item())
            valid_correct = float(pred_valid.eq(data.y.squeeze()[valid_begin:test_begin]).sum().item())

            print("Train Turn %d:" % (i + 1))
            print("Acc on Train is %f, Acc on Valid is %f" % (train_correct / valid_begin, valid_correct / (test_begin - valid_begin)))

            del pred_train, pred_valid, train_correct, valid_correct, _
            torch.cuda.empty_cache()

        del train_out, valid_out
        torch.cuda.empty_cache()

    model.eval()

    # 下面在测试集进行验证
    _, pred_test = model(data)[test_begin:LABEL_LEN, :].max(dim=1)
    test_correct = float(pred_test.eq(data.y.squeeze()[test_begin:]).sum().item())
    print("Acc on Test = %f" % (test_correct / (LABEL_LEN - test_begin)))

    del _, pred_test, test_correct
    torch.cuda.empty_cache()

    torch.save(model, "model")


