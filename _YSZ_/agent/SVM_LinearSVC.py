from sklearn import svm
from sklearn.pipeline import make_pipeline


class LinearSVC():
    """Decision Tree: CART"""

    def __init__(self, probability=False):
        super(LinearSVC, self).__init__()
        self.model = svm.SVC(kernel='linear', probability=probability)

    def save_to_desk(self, path, file_name):
        """存储模型"""
        import pickle
        with open(path + file_name + ".pickle", 'wb') as f:
            pickle.dump(self.model, f)

    def load_from_desk(self, path, file_name):
        """加载模型"""
        import pickle
        with open(path + file_name + ".pickle", 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, X, y):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        self.model.fit(X, y)
        return self

    def forward(self, x, train=False):
        """x is [[1]]"""
        result = self.model.predict([x])[0]
        return result

    def forward_proba(self, x, train=False):
        """x is [[1]]"""
        result = self.model.predict_proba([x])[0]
        return result
