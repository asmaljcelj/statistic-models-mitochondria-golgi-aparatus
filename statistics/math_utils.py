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
            base_vector[0] * base_vector[2] * (1 - math.cos(angle)) + base_vector[1] * math.sin(angle)
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


def rotate_vector_test(v1, v2, v3, current_vector, angle):
    theta = np.radians(angle)
    R = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ]
    rotated = np.dot(R, current_vector)


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def from_degrees_to_radians(degress):
    return math.pi * degress / 180.0
