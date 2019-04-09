import matplotlib.pyplot as plt
import numpy as np



def two_scale_plot(y1, y2, y1_label= 'y1', y2_label = 'y2', x=None, xlabel='Iteration'):
    '''

    :param y_1:
    :param y_2:
    :param x:
    :return:
    '''

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label, color=color)
    # ax1.set_ylim((0,15))
    if x is not None:
        ax1.plot(x, y1, color=color)
    else:
        ax1.plot(y1, color=color)




    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel(y2_label)

    if x is not None:
        ax2.plot(x, y2, color=color)
    else:
        ax2.plot(y2, color=color)
    ax2.tick_params(axis='y', labelcolor = color)


