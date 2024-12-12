import os

import numpy as np
from matplotlib import pyplot as plt



data_directory = '../ga_instances'

def plot_instance(ga_instance):
    for d in ga_instance:
        cisternae = ga_instance[d]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cisternae[:, 0], cisternae[:, 1], cisternae[:, 2])
        plt.show()


for filename in os.listdir(data_directory):
    data = np.load(data_directory + '/' + filename, allow_pickle=True)
    plot_instance(data)
