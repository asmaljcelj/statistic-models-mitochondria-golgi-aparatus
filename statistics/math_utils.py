import csv
import math

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


def distance_between_points(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def rotate_vector(vector, angle, base_vector):
    R = [
        [
            math.cos(angle) + base_vector[0] ** 2 * (1 - math.cos(angle)),
            base_vector[0] * base_vector[1] * (1 - math.cos(angle)) - base_vector[2] * math.sin(angle),
            base_vector[0] * base_vector[1] * (1 - math.cos(angle) + base_vector[1] * math.sin(angle))
        ],
        [
            base_vector[1] * base_vector[0] * (1 - math.cos(angle)) + base_vector[2] * math.sin(angle),
            math.cos(angle) + base_vector[1] ** 2 * (1 - math.cos(angle)),
            base_vector[1] * base_vector[2] * (1 - math.cos(angle)) - base_vector[0] * math.sin(angle)
        ],
        [
            base_vector[2] * base_vector[0] * (1 - math.cos(angle)) - base_vector[1] * math.sin(angle),
            base_vector[2] * base_vector[1] * (1 - math.cos(angle)) + base_vector[0] * math.sin(angle),
            math.cos(angle) + base_vector[2] ** 2 * (1 - math.cos(angle))
        ]
    ]
    R = np.array(R)
    return np.dot(R, vector)
