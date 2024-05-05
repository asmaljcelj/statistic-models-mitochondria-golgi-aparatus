import csv

import numpy as np
from scipy.special import binom


def magnitude(point):
    return np.sqrt(np.sum(point ** 2))


def extract_points(csv_file):
    points = []
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        points.append([int(row[0]), int(row[1]), int(row[2])])
    return points


def calculate_B(n, i, t):
    return binom(n, i) * t ** i * (1 - t) ** (n - i)