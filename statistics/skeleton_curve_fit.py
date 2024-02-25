from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import csv
import numpy as np


def function(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y


file_path = '../skeletons/fib1-0-0-0_5.csv'

x, y, z = [], [], []

with open(file_path) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        x.append(int(row[0]))
        y.append(int(row[1]))
        z.append(int(row[2]))

popt, pcov = curve_fit(function, (x, y), z)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue')
x_range = np.linspace(min(x), max(x), 1000)
y_range = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(x_range, y_range)
Z = function((X, Y), *popt)
ax.plot_surface(X, Y, Z, color='red', alpha=0.5)
# ax.view_init(20, 30)
plt.show()



