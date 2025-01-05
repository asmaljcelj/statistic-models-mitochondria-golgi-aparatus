import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gradient = [
 [0.0, 0.0, 1.0, 1.0],
 [0.0706, 0.0, 0.9255, 1.0],
 [0.1412, 0.0, 0.8549, 1.0],
 [0.2118, 0.0, 0.7843, 1.0],
 [0.2824, 0.0, 0.7137, 1.0],
 [0.3569, 0.0, 0.6392, 1.0],
 [0.4275, 0.0, 0.5686, 1.0],
 [0.4980, 0.0, 0.4980, 1.0],
 [0.5686, 0.0, 0.4275, 1.0],
 [0.6392, 0.0, 0.3569, 1.0],
 [0.7137, 0.0, 0.2824, 1.0],
 [0.7843, 0.0, 0.2118, 1.0],
 [0.8549, 0.0, 0.1412, 1.0],
 [0.9255, 0.0, 0.0706, 1.0],
 [1.0, 0.0, 0.0, 1.0]
]

data_directory = '../ga_instances'

def plot_instance(ga_instance):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = Axes3D(fig)
    for i, d in enumerate(ga_instance):
        cisternae = ga_instance[d]
        ax.scatter(cisternae[:, 0], cisternae[:, 1], cisternae[:, 2], c=[[gradient[i % len(gradient)]]], label=str(i))
    # ax.legend()
    plt.show()


def calculate_statistics_on_instance(ga_instance):
    data = {}
    for d in ga_instance:
        statistical_data = calculate_statistics(ga_instance[d])

def calculate_statistics(cistanae):
    # todo: thickness, shape
    # https: // www.cad - journal.net / files / vol_13 / CAD_13(2)_2016_199 - 207.pdf
    pass


for filename in os.listdir(data_directory):
    data = np.load(data_directory + '/' + filename, allow_pickle=True)
    plot_instance(data)
