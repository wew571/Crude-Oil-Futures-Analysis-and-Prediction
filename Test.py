import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

import decision_tree_regression
import correlation_matrix
import hist_crude_oil

df = pandas.read_csv('Crude_Oil_Futures.csv')

print(correlation_matrix.corr(df , 'spearman'))

plt.figure(figsize=(5 , 5))
sns.heatmap(correlation_matrix.corr(df , 'spearman') , annot=True , cmap = 'coolwarm')
plt.show()
X_train = numpy.array([[1], [2], [3], [4], [5], [6]])
y_train = numpy.array([1.1, 1.9, 3.0, 4.1, 4.9, 6.0])


# Train cây
tree = decision_tree_regression.DecisionTree(max_depth=2)
tree.fit(X_train, y_train)

# Test dự đoán
X_test = numpy.array([[1.5], [3.5], [5.5]])
y_pred = tree.predict(X_test)

print("Dự đoán:", y_pred)

hist_crude_oil.draw()
