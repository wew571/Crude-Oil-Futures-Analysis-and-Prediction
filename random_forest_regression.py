import numpy

import decision_tree_regression

class RandomForestRegression:
    def __init__(self , n_tree = 200 , max_depth = 25 , min_split = 2 , n_feature = None):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_split = min_split
        self.n_feature = n_feature
        self.trees = []

    def bootstrap(self , X ,y ):
        """ Chọn ngẫu nhiên 1 phần tử trong 1 hàng"""
        n_samples = X.shape[0]
        index = numpy.random.choice(n_samples , n_samples , replace = True)
        return X[index] , y[index]

    def fit(self , X , y ):
        """ Build rừng cây """
        self.trees = []
        for _ in range(self.n_tree):
            tree = decision_tree_regression.DecisionTree(
                max_depth = self.max_depth,
                min_split = self.min_split
            )

            X_sample , y_sample = self.bootstrap(X , y)
            tree.fit(X_sample , y_sample)
            self.trees.append(tree)

    def predict(self , X ):
        """ Hàm dự đoán kết quả cuối cùng"""
        predictions = numpy.array([tree.predict(X) for tree in self.trees])
        return numpy.mean(predictions ,  axis=0)
