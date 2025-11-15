import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer

import decision_tree_regression
import correlation_matrix
import hist_crude_oil
import random_forest_regression

df = hist_crude_oil.get_data()

# X_train = numpy.array([[1], [2], [3], [4], [5], [6]])
# y_train = numpy.array([1.1, 1.9, 3.0, 4.1, 4.9, 6.0])

Xr_train = df['open'].to_numpy().reshape(-1,1)

yr_train = df['close'].to_numpy()

start = default_timer()
rf = random_forest_regression.RandomForestRegression(max_depth=9)
rf.fit(Xr_train, yr_train)

yr_pred = rf.predict(Xr_train)
finish = default_timer()

print(rf.r2_score(yr_train, yr_pred))
print(finish - start)
