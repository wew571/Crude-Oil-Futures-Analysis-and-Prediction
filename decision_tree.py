import pandas
import numpy

class Node:
    def __init__(self , value , feature_index = None , threshold = None , left = None, right = None):
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self , max_depth = 5 , min_split = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.root = None

    def gini_impurity(self , y) -> float:
        """
        Tính Gini Impurity cho 1 node

        Parameters:
            y : array-like

        Returns:
            value : float

        Công thức toán học :

            G = 1 - Σ[P_i²]

            G       Geni Impurity
            P_i     Xác suất
        """
        if len(y) == 0:
            return 0
        count = numpy.bincount(y)
        probability = count / len(y)

        value = 1 - numpy.sum(probability**2)
        return value

    def entropy(self , y) -> float:
        """
        Tính Entropy cho 1 node:

        Parameters:
            y : array-like

        Returns:
            value : float

        Công Thức Toán Học:

            E = - Σ[ P_i x log2(P_i) ]

            E       Entropy
            P_i     Xác suất
        """
        if len(y) == 0:
            return 0
        count = numpy.bincount(y)
        probability = count / len(y)

        probability = probability[probability > 0]
        value = -1 * numpy.sum(probability * numpy.log2(probability))
        return value

    def mse(self , y) -> float:
        """
        Tính MSE ( Mean Square Error ) cho 1 node:

        Parameters:
            y : array-like

        Returns:
            value : float

        Công Thức Toán Học:
                   1
            MSE = --- Σ[ (y - ȳ)² ]
                   n

            ȳ   giá trị trung bình của y
        """
        if len(y) == 0:
            return 0
        value = numpy.mean((y - y.mean()) ** 2)
        return value

    def find_best_split_regression(self ,X , y):
        """
        Tìm split tốt nhất cho regression

        Parameters:
            X : feature
            y : label

        Returns:

        Công Thức Toán Học:

            MSE(r) = MSE(p) -  MSE(c)


            MSE(r)      MSE reduction
                                                   1
            MSE(p)      MSE parent      MSE(p) = ----- Σ[ (y - ȳ)² ]
                                                   n

                                                   nL                          nR
            MSE(c)      MSE child       MSE(c) = ------ Σ[ (yL - ȳL)² ]  +  ------- Σ[ (yR - ȳR)² ]
                                                    n                           n
        """

        best_mse_reduction = -1
        best_threshold = None
        best_feature = None

        n , n_col = X.shape
        mse_parent = self.mse(y)

        for feature_index in range(n_col):
            feature_value = X[: , feature_index]

            for threshold in feature_value:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if numpy.sum(left_mask) < self.min_split or numpy.sum(right_mask) < self.min_split:
                    continue

                mse_left = self.mse(y[left_mask])
                mse_right = self.mse(y[right_mask])

                n_left = len(y[left_mask])
                n_right = len(y[right_mask])

                mse_children = n_left/n * mse_left + n_right/n * mse_right
                mse_reduction = mse_parent - mse_children

                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_threshold = threshold
                    best_feature = feature_index

        return best_mse_reduction , best_threshold , best_feature









