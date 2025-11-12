import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import correlation_matrix

df = pd.read_csv('Crude_Oil_Futures.csv')

print(correlation_matrix.corr(df , 'spearman'))

plt.figure(figsize=(5 , 5))
sns.heatmap(correlation_matrix.corr(df , 'spearman') , annot=True , cmap = 'coolwarm')
plt.show()




