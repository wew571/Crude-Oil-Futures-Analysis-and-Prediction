import numpy

import decision_tree_regression

class RandomForestRegression:
    def __init__(self , n_tree = 25 , max_depth = 9 , min_split = 2 , n_feature = None):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_split = min_split
        self.n_feature = n_feature
        self.trees = []
        self.feature_selection = []

    def bootstrap(self , X ,y ):
        """ Chọn ngẫu nhiên 1 phần tử trong 1 hàng"""
        n_samples = X.shape[0]
        index = numpy.random.choice(n_samples , n_samples , replace = True)
        return X[index] , y[index]

    def random_feature_selection(self , n_feature):
        """Chọn ngẫu nhiên tập feature con"""
        if self.n_feature is None:
            n_selection = int(numpy.sqrt(n_feature))
        else:
            n_selection = min(self.n_feature , n_feature)

        return numpy.random.choice(n_feature , n_selection , replace = False)

    def fit(self , X , y ):
        """ Build rừng cây """
        self.trees = []
        self.feature_selection = []
        n_feature = X.shape[1]

        for _ in range(self.n_tree):
            tree = decision_tree_regression.DecisionTree(
                max_depth = self.max_depth,
                min_split = self.min_split
            )

            X_sample , y_sample = self.bootstrap(X , y)
            selected_feature = self.random_feature_selection(n_feature)
            X_sample_selected = X_sample[: , selected_feature]

            tree.fit(X_sample_selected , y_sample)
            self.trees.append(tree)
            self.feature_selection.append(selected_feature)

    def predict(self , X ):
        """ Hàm dự đoán kết quả cuối cùng"""
        predictions = []
        for i , tree in enumerate(self.trees):
            selected_feature = self.feature_selection[i]
            X_selected = X[: , selected_feature]
            pred = tree.predict(X_selected )
            predictions.append(pred)

        predictions = numpy.array(predictions)
        return numpy.mean(predictions ,  axis=0)

    def r2_score(self , y , y_pred)-> float:
        """
        Hàm đánh giá độ chính xác của dữ liệu đầu ra

        Tổng quan về kết quả đầu ra :
            -Trong biến r này sẽ có kết quả trong khoảng [ 0 , 1]
            -Điều đó có nghĩa là nếu cái r² này nó mà ra trong khoảng từ [ 0.1 -> 0.4 ]
            thì có nghĩa là mô hình này có khả năng đoán sai rất cao . Còn trong khoảng
            [0.5] thì tỷ lệ đoán đúng cũng chỉ có 50% . Nếu muốn đoán toàn chuẩn thì ít
            nhất cũng phải từ [0.9 -> 0.99] thì may ra

        Về công thức toán học được sử dụng :
                    Σ[(y - y_pred)²]
        r² = 1 -  ------------------
                    Σ[(y - ȳ)²]

                        SS_res
        <=> r² = 1 -  ------------
                        SS_tot

        y_perd      giá trị y dự đoán
        SS_res      Residual Sum of Squares ( Tổng bình phương phần dư )
        SS_tot      Total Sum of Squares    ( Tổng bình phương sai lệch so với trung bình )
        """
        y = y.astype(float)
        y_pred = y_pred.astype(float)
        ss_res = numpy.sum(( y - y_pred) ** 2)
        ss_tot = numpy.sum((y - numpy.mean(y))**2)
        return 1 - ss_res/ss_tot
