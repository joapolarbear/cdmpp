import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {"color": "darkred",
        "size": 13,
        "family": "serif"}

def plot_xgb_history(results):
    ''' Visualize the training process for XGB'''
    fig, axs = plt.subplots(3, 1)
    ax0: matplotlib.axes.Axes = axs[0]
    ax1: matplotlib.axes.Axes = axs[1]

    x = np.arange(0, 1000, 1)

    ax0.plot(x, results['Train']['MAPE'], label='train-MAPE')
    ax0.plot(x, results['Test']['MAPE'], label='test-MAPE')
    ax0.legend()

    ax1.plot(x, results['Train']['rmse'], label='train-rmse')
    ax1.plot(x, results['Test']['rmse'], label='test-rmse')
    ax1.legend()
    plt.savefig("tmp.png")