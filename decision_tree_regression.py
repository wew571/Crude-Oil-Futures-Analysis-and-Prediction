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
    def __init__(self , max_depth = 25 , min_split = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.root = None

    def mse(self , y) -> float:
        """
        Tính MSE ( Mean Square Error ) cho 1 node:

        Parameters:
            y : target value

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
            y : target value

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
                # Chia dữ liệu thành 2 nửa
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if numpy.sum(left_mask) < self.min_split or numpy.sum(right_mask) < self.min_split:
                    continue

                mse_left = self.mse(y[left_mask])
                mse_right = self.mse(y[right_mask])

                n_left = len(y[left_mask])
                n_right = len(y[right_mask])

                # Tính MSE_children (MSE(c)) avf MSE_reduction(MSE(r))
                mse_children = n_left/n * mse_left + n_right/n * mse_right
                mse_reduction = mse_parent - mse_children

                # Tìm MSE_reduction tốt nhất
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_threshold = threshold
                    best_feature = feature_index

        return best_feature , best_threshold , best_mse_reduction

    def build_tree(self , X , y , depth = 0):
        """Build cây quyết định"""
        n = len(y)
        if depth >= self.max_depth or n < self.min_split:
            return Node(value=numpy.mean(y))

        feature , threshold , mse_reduction = self.find_best_split_regression(X , y)

        # ---- Giải thích tại sao lại là value=numpy.mean(y)-----
        # Theo công thức tính MSE sẽ là :
        #                       1
        #            MSE    = ----- Σ[ (y - ȳ)² ]
        #                       n
        # Đặt ȳ là c :
        #                       1
        #            MSE(c) = ----- Σ[ (y - c)² ]
        #                       n
        #
        #            => L(c) = Σ[ y² - 2yc + c² ]
        #            -> L(c) = Σ[y²] - 2cΣ[y] + nc²         { Σc² = nc² }
        # Để c có nghiệm thì phải đạo hàm MSE(c) theo c :
        #            L'(c) = -2Σ[y] + 2nc
        #
        # Cho đạo hàm bằng 0 để có nghiệm :
        #            nc - Σ[y] = 0
        #                      1
        #            <=> c = ------ Σ[y]
        #                      n
        #  Đây chính là công thức của trung bình cộng nha
        if feature is None or mse_reduction <= 0:
            return Node(value=numpy.mean(y))

        # chia dữ liệu thành 2 nửa
        left_mask = X[ : , feature] <= threshold
        right_mask = ~left_mask

        # Build nhánh cây
        left = self.build_tree(X[left_mask] , y[left_mask] , depth = depth + 1)
        right = self.build_tree(X[right_mask] , y[right_mask] , depth = depth + 1)

        return Node(
                    value = None ,
                    feature_index = feature ,
                    threshold = threshold ,
                    left = left ,
                    right = right)

    def fit(self , X , y):
        """ Dùng để Train mô hình """
        self.root = self.build_tree( X , y)

    def predict_one(self , X , node = None):
        """ Duyệt cây """
        if node is None:
            node = self.root

        while node.feature_index is not None:
            if X[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self , X ):
        value = numpy.array([self.predict_one(x) for x in X])
        return value

