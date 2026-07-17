from sklearn.ensemble import GradientBoostingClassifier as GBC


class DT_GBDT():
    """Decision Tree: CART"""

    def __init__(self, random_state=10):
        super(DT_GBDT, self).__init__()
        self.model = GBC(random_state=random_state)

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
        result_proba = self.model.predict_proba([x])[0]
        return result_proba
