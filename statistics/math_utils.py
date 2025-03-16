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


def extract_points(csv_file):
    points = []
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        points.append([int(row[0]), int(row[1]), int(row[2])])
    return points


def calculate_B(n, i, t):
    return binom(n, i) * t ** i * (1 - t) ** (n - i)


def distance_between_points(point1, point2):
    result = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
    return round(result, 2)


def rotate_vector(vector, angle_degrees, base_vector):
    # Source: Rodrigues' rotation formula
    theta = np.deg2rad(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    term1 = vector * cos_theta
    term2 = np.cross(base_vector, vector) * sin_theta
    term3 = base_vector * np.dot(base_vector, vector) * (1 - cos_theta)
    return term1 + term2 + term3


def rotate_vector2(v, R):
    return np.dot(R, v)


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


# Step 1: Rotate the normal n1 to align with n2
def align_normals(v, n1, n2):
    if np.allclose(n1, n2):
        return v, None  # No rotation needed if normals are already aligned

    axis = np.cross(n1, n2)
    angle = angle_between(n1, n2)
    R = rotation_matrix_from_axis_angle(axis, angle)
    return rotate_vector2(v, R), rotate_vector2(n1, R)


def rotation_matrix_from_axis_angle(axis, angle):
    axis = normalize(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_prod_matrix = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
    identity_matrix = np.identity(3)
    return cos_theta * identity_matrix + sin_theta * cross_prod_matrix + (1 - cos_theta) * np.outer(axis, axis)


# Step 2: Rotate around the aligned normal to match o1 with o2
def align_orthogonal(v, o1, o2, normal):
    # Project o1 and o2 onto the plane perpendicular to the normal
    o1_proj = o1 - np.dot(o1, normal) * normal
    o2_proj = o2 - np.dot(o2, normal) * normal

    if np.linalg.norm(o1_proj) == 0 or np.linalg.norm(o2_proj) == 0:
        return v  # If projections are zero, no further rotation is needed

    o1_proj = normalize(o1_proj)
    o2_proj = normalize(o2_proj)

    axis = normal  # Rotation around the normal
    angle = angle_between(o1_proj, o2_proj)
    R = rotation_matrix_from_axis_angle(axis, angle)
    return rotate_vector2(v, R)


# Function to rotate a vector using a rotation matrix
def rotate_vector1(original_vector, original_normal, new_normal, original_orthogonal, new_orthogonal):
    v_aligned, n1_aligned = align_normals(original_vector, original_normal, new_normal)
    return align_orthogonal(v_aligned, original_orthogonal, new_orthogonal, new_normal)

# Function to compute the angle between two vectors
def angle_between(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def get_rotation_arbitrary_matrix():
    pass


def get_points_between_2_points(point1, point2, num_of_points):
    points = []
    mx = point2[0] - point1[0]
    my = point2[1] - point1[1]
    mz = point2[2] - point1[2]
    if num_of_points == 0:
        num_of_points = 1
    for i in np.arange(0, 1.01, 1 / num_of_points):
        x = point1[0] + mx * i
        y = point1[1] + my * i
        z = point1[2] + mz * i
        points.append([x, y, z])
    return points


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


def new_point(theta, phi, base_x, base_y, base_z):
    base_coords = spherical_to_cartesian(1, theta, phi)
    new_coords = transform_to_new_axes(base_coords, base_x, base_y, base_z)
    return transform_to_base(new_coords, base_x, base_y, base_z)


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def transform_to_new_axes(cartesian_coords, ux, uy, uz):
    transformation_matrix = np.array([ux, uy, uz]).T
    return transformation_matrix.dot(cartesian_coords)


def transform_to_base(cartesian_coords_in_new_axes, ux, uy, uz):
    transformation_matrix = np.array([ux, uy, uz]).T
    # Compute the inverse of the transformation matrix
    inverse_matrix = np.linalg.inv(transformation_matrix)
    return inverse_matrix.dot(cartesian_coords_in_new_axes)


def plot_histogram(data):
    # Plotting a basic histogram
    plt.hist(data, bins=15, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')

    # Display the plot
    plt.show()


def calculate_average_and_standard_deviation(data):
    average = np.average(data, axis=0)
    summed = 0
    for point in data:
        summed += np.square(average - point)
    standard_deviation = np.sqrt(summed / (len(data) - 1))
    return average, standard_deviation


def distance_between_point_and_line(point, line_vector):
    point = point.astype(np.float64)
    line_point = np.array([0, 0, 0])
    ap = point - line_point
    cross = np.cross(ap, line_vector)
    magnitude = np.linalg.norm(cross)
    direction_magnitude = np.linalg.norm(line_vector)
    return magnitude / direction_magnitude


def calculate_average_cisterna(cisterna):
    return [sum(sublist) / len(sublist) for sublist in cisterna]


def generate_direction_vectors(n=8):
    vectors = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = math.cos(angle)
        y = math.sin(angle)
        vectors.append([0, y, x])
    return vectors


def get_voxel_on_plane(voxels, point, normal, epsilon=1e-6):
    coordinates = np.argwhere(voxels == 1)
    x0, y0, z0 = point
    A, B, C = normal
    distances = A * (coordinates[:, 0] - x0) + B * (coordinates[:, 1] - y0) + C * (coordinates[:, 2] - z0)
    return coordinates[np.abs(distances) < epsilon]
