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
    def __int__(self , max_depth = 5 , min_split = 2):
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
        value = numpy.mean(y - y.mean() ** 2)
        return value









