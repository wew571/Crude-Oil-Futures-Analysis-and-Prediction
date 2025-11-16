import numpy
import pandas
from matplotlib import pyplot as plt
import seaborn


def draw_pair_plot(df, show_plot=True):
    """Vẽ pair plot, trả về figure nếu show_plot=False"""
    pair_grid = seaborn.pairplot(df[['open', 'close', 'high', 'low', 'volume']])
    plt.xlabel('Open')
    plt.ylabel('Close')

    if show_plot:
        plt.show()
    else:
        return pair_grid.fig


def draw_prediction_plot(y_true, y_pred, show_plot=True):
    """Vẽ prediction plots, trả về figure nếu show_plot=False"""
    if show_plot:
        # 1. Scatter plot True vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted (Scatter)')
        plt.grid(True, alpha=0.3)
        plt.show()

        # 2. Line plot True & Predicted
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="True Close Price")
        plt.plot(y_pred, label="Predicted Price")
        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Close Price")
        plt.title("Close Price Prediction")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # Trả về figure tổng hợp cho tkinter
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('True vs Predicted (Scatter)')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Line plot
        sample_indices = range(min(50, len(y_true)))
        ax2.plot(sample_indices, y_true[:50], 'b-', label="True Close Price", linewidth=2)
        ax2.plot(sample_indices, y_pred[:50], 'r-', label="Predicted Price", linewidth=2)
        ax2.legend()
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Close Price")
        ax2.set_title("Close Price Prediction")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
