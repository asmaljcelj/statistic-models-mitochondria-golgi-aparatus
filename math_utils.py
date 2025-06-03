import csv
import math

import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def magnitude(point):
    return np.sqrt(np.sum(point ** 2))


def normalize(vector):
    return vector / magnitude(vector)


def calculate_B(n, i, t):
    return binom(n, i) * t ** i * (1 - t) ** (n - i)


def distance_between_points(point1, point2):
    result = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
    return round(result, 2)


def rotate_vector(vector, angle_degrees, base_vector):
    # Rodrigues' rotation formula
    theta = np.deg2rad(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    term1 = vector * cos_theta
    term2 = np.cross(base_vector, vector) * sin_theta
    term3 = base_vector * np.dot(base_vector, vector) * (1 - cos_theta)
    return term1 + term2 + term3


def are_opposite_vectors(v1, v2):
    dot = np.dot(v1, v2)
    return np.isclose(dot, -np.linalg.norm(v1) * np.linalg.norm(v2))


# source: https://math.stackexchange.com/a/476311
def get_rotation_matrix(origin, destination):
    v = np.cross(origin, destination)
    if not np.any(v):
        identity = np.eye(3)
        if are_opposite_vectors(origin, destination):
            identity[2] *= -1
        return identity
    s = np.linalg.norm(v)
    c = np.dot(origin, destination)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
    return r


def random_cosine(u, v, m, skeleton_angle_increment=1):
    theta = np.arccos(np.power(1 - u, 1 / (1 + m)))
    phi = 2 * np.pi * v
    # dodaj tocke na kroznici (z = 0)
    num_of_points = int(360 / skeleton_angle_increment)
    theta = np.append([np.pi / 2] * num_of_points, theta, axis=0)
    phi = np.append([np.deg2rad(degree) for degree in range(0, 360, skeleton_angle_increment)], phi, axis=0)
    # Switch to cartesian coordinates
    x = np.round(np.sin(theta) * np.cos(phi), 6)
    y = np.round(np.sin(theta) * np.sin(phi), 6)
    z = np.round(np.cos(theta), 6)
    coordinate_angle_dict = {(x_coord, y_coord, z_coord): (t, p) for x_coord, y_coord, z_coord, t, p in zip(x, y, z, theta, phi)}
    return np.array(list(zip(x, y, z))), coordinate_angle_dict


def frenet_serre(matrix, t, curvature, torsion):
    # if torsion != 0:
    #     print()
    gamma_prime1 = matrix[3]
    gamma_prime2 = matrix[4]
    gamma_prime3 = matrix[5]

    t_prime1 = curvature * matrix[6]
    t_prime2 = curvature * matrix[7]
    t_prime3 = curvature * matrix[8]

    n_prime1 = -curvature * matrix[3] + torsion * matrix[9]
    n_prime2 = -curvature * matrix[4] + torsion * matrix[10]
    n_prime3 = -curvature * matrix[5] + torsion * matrix[11]

    b_prime1 = -torsion * matrix[6]
    b_prime2 = -torsion * matrix[7]
    b_prime3 = -torsion * matrix[8]

    return [
        gamma_prime1, gamma_prime2, gamma_prime3,
        t_prime1, t_prime2, t_prime3,
        n_prime1, n_prime2, n_prime3,
        b_prime1, b_prime2, b_prime3
    ]


def calculate_next_skeleton_point(last_skeleton_point, T, N, B, curvature, torsion, distance_to_next_point):
    matrix_to_solve = [
        last_skeleton_point[0], last_skeleton_point[1], last_skeleton_point[2],
        T[0], T[1], T[2],
        N[0], N[1], N[2],
        B[0], B[1], B[2]
    ]
    t = np.array([0, distance_to_next_point])
    result = odeint(frenet_serre, matrix_to_solve, t, args=(curvature, torsion))
    return result


def calculate_average_and_standard_deviation(data):
    average = np.average(data, axis=0)
    summed = 0
    for point in data:
        summed += np.square(average - point)
    standard_deviation = np.sqrt(summed / (len(data) - 1))
    return average, standard_deviation


def generate_direction_vectors(base, n=8):
    vectors = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        direction = np.cos(angle) * base[0] + np.sin(angle) * base[1]
        # x = math.cos(angle)
        # y = math.sin(angle)
        # vectors.append(base.T @ [0, y, x])
        vectors.append(normalize(direction))
    return vectors
