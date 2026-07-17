from sklearn.neighbors import KNeighborsClassifier
import numpy


class huigui():
    """KNN KD-Tree"""

    def __init__(self, action_count=-1, function="linear"):
        super(huigui, self).__init__()
        self.ACTION_COUNT = action_count
        from sklearn import neighbors
        from sklearn import ensemble
        from sklearn.tree import ExtraTreeRegressor
        from sklearn import tree
        from sklearn import linear_model
        from sklearn import svm
        # self.model_list = [ExtraTreeRegressor() for i in range(action_count)]
        # self.model_list = [neighbors.KNeighborsRegressor(n_neighbors=100) for i in range(action_count)]
        # self.model_list = [neighbors.KNeighborsRegressor(n_neighbors=1) for i in range(action_count)]
        self.model_list = [ensemble.RandomForestRegressor() for i in range(action_count)]

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

    def fit(self, Xs, ys):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        for i in range(len(self.model_list)):
            self.model_list[i].fit(Xs[i], ys[i])
        return self

    def forward(self, x, train=False):
        """x is [[1]]"""
        results = []
        for i in range(self.ACTION_COUNT):
            results.append(self.model_list[i].predict([x])[0])
        max = numpy.argmax(results)
        return max

    def forward_qs(self, x, train=False):
        """x is [[1]]"""
        results = []
        for i in range(self.ACTION_COUNT):
            results.append(self.model_list[i].predict([x])[0])
        return results