import numpy as np


class KNN_Force():
    """KNN Force searching"""

    def __init__(self):
        super(KNN_Force, self).__init__()

    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def save_to_desk(self, path, file_name):
        """存储模型"""
        import pickle
        with open(path + file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

    def load_from_desk(self, path, file_name):
        """加载模型"""
        import pickle
        with open(path + file_name + ".pickle", 'rb') as f:
            self = pickle.load(f)

    def fit(self, X, Y):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        self.X = X
        self.Y = Y
        self.Exp_Count = len(X)
        return self

    def forward(self, x, train=False):
        """x is [[1]]"""
        nearest = [-1, float('inf')]
        for i in range(self.Exp_Count):
            cur_dis = self.distance(x, self.X[i])
            if cur_dis < nearest[1]:
                nearest[0] = i
                nearest[1] = cur_dis
        result = self.Y[nearest[0]]
        return result

    def forward_sorted_proba(self, x, train=False):
        labels = []
        dis = []
        # 遍历所有点，找到每个类别的最近距离
        for i in range(self.Exp_Count):
            cur_dis = self.distance(x, self.X[i])
            cur_label = self.Y[i]
            if cur_label in labels:  # 不存在，添加
                if cur_dis < dis[labels.index(cur_label)]:  # 如果距离更小
                    dis[labels.index(cur_label)] = cur_dis
            else:  # 存在，更新
                labels.append(cur_label)
                dis.append(cur_dis)
        # 排序
        sorted_dis = sorted(range(len(dis)), key=lambda k: dis[k], reverse=False)
        result = []
        for idx in sorted_dis:
            result.append(labels[idx])
        return result
