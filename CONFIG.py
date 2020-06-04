"""
这个文件定义了一些全局需要使用的数据定义。
比如，图的大小，等关键数据。
"""


class CONFIG:
    @staticmethod
    def FeatureLen():
        """
        :return: 返回特征向量的长度（也即是维度）
        """
        return 100

    @staticmethod
    def ClassNum():
        """
        :return: 返回类别的总数，根据OpData中的测试代码得到，最小Label=0，最大Label=17
        """
        return 18

    @staticmethod
    def TargetBegin():
        return 543486

    @staticmethod
    def TargetEnd():
        return 593486

    @staticmethod
    def AttackRate():
        return 1
