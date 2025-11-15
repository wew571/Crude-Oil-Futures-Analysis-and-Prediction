import numpy
import pandas
from matplotlib import pyplot
import seaborn

import hist_crude_oil

df = hist_crude_oil.get_data()

def draw_pair_plot(df):
    seaborn.pairplot(df[['open' , 'close' , 'high' , 'low' , 'volume' ]])
    pyplot.xlabel('Open')
    pyplot.ylabel('Close')
    pyplot.show()

def draw_prediction_plot(y_true, y_pred):
    # 1. Scatter plot True vs Predicted
    pyplot.scatter(y_true, y_pred)
    pyplot.xlabel('True Values')
    pyplot.ylabel('Predicted Values')
    pyplot.title('True vs Predicted (Scatter)')
    pyplot.show()

    # 2. Line plot True & Predicted
    pyplot.plot(y_true, label="True Close Price")
    pyplot.plot(y_pred, label="Predicted Price")
    pyplot.legend()
    pyplot.xlabel("Sample Index")
    pyplot.ylabel("Close Price")
    pyplot.title("Close Price Prediction")
    pyplot.show()

