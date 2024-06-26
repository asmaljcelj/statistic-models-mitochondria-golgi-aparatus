import os

import numpy as np

from bezier import perform_arc_length_parametrization_bezier_curve, calculate_bezier_derivative, calculate_bezier_second_derivative
from math_utils import magnitude, rotate_vector, distance_between_points, normalize, get_rotation_matrix
from outside_statistics import calculate_average, group_distances, sample_new_points
from utils import read_file_collect_points, read_nii_file, plot_new_points, save_as_nii


def sample_direction_vectors(num_of_samples):
    u_samples = np.random.uniform(0, 1, num_of_samples)
    v_samples = np.random.uniform(0, 1, num_of_samples)
    return random_cosine(u_samples, v_samples, 1)


def random_cosine(u, v, m):
    theta = np.arccos(np.power(1 - u, 1 / (1 + m)))
    phi = 2 * np.pi * v

    # Switch to cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array(list(zip(x, y, z)))


def uniform_sample_at_ends(end_point, second_point, num_of_samples, shape, points_on_hemisphere, skeleton, parametrized_points):
    # u_samples = np.random.uniform(0, 1, num_of_samples)
    # v_samples = np.random.uniform(0, 1, num_of_samples)
    # points_on_hemisphere = random_cosine(u_samples, v_samples, 1)
    normal = normalize(end_point - second_point)
    R = get_rotation_matrix(np.array([0, 0, 1]), normal)
    sampled_points_t = {}
    distances = {}
    # rotated_vectors = []
    for point in points_on_hemisphere:
        direction_vector = normalize(point)
        rotated_vector = np.dot(R, direction_vector)
        # rotated_vectors.append(rotated_vector)
        distance, sampled_points = iterate_ray(end_point, rotated_vector, shape)
        key = (direction_vector[0], direction_vector[1], direction_vector[2])
        distances[key] = distance
        sampled_points_t[key] = sampled_points
    # plot_sampling_with_shape(shape, sampled_points_t, skeleton, parametrized_points)
    # rotated_vectors = np.array(rotated_vectors)
    # plot_new_points(rotated_vectors)
    return distances


# todo: neki cudno sampla tudi izven BB-ja???????? (treba popravit)
def sample_rays(origin, direction_vector, shape, base_x, base_y, parametrized_points, skeleton):
    original_direction_vector = direction_vector
    distances = []
    sampled_points = {}
    base_x = base_x / magnitude(base_x)
    base_y = base_y / magnitude(base_y)
    counter = 0
    while not np.allclose(direction_vector, original_direction_vector) or counter == 0:
        distance, sampled_points_a = iterate_ray(origin, direction_vector, shape)
        distances.append(distance)
        sampled_points[counter] = sampled_points_a
        direction_vector = get_new_direction_vector(original_direction_vector, base_x, counter)
        counter += 1
    # plot_sampling_with_shape(shape, sampled_points, skeleton, parametrized_points)
    return distances


def iterate_ray(origin, direction_vector, shape):
    current_point = origin
    sampled_points_a = [current_point]
    previous_voxel, current_voxel = None, [int(origin[0]), int(origin[1]), int(origin[2])]
    while True:
        new_point = current_point + direction_vector
        x_coord, y_coord, z_coord = int(new_point[0]), int(new_point[1]), int(new_point[2])
        previous_voxel = current_voxel
        current_voxel = [x_coord, y_coord, z_coord]
        in_boundary = x_coord < shape.shape[0] and y_coord < shape.shape[1] and z_coord < shape.shape[2] and \
                      shape[x_coord][y_coord][z_coord] > 0
        # todo: improved collision detection and distance measurement
        if not in_boundary:
            # distance += magnitude(direction_vector)
            boundary_voxel = [previous_voxel[0], previous_voxel[1], previous_voxel[2]]
            distance = distance_between_points(origin, boundary_voxel)
            break
        sampled_points_a.append(new_point)
        current_point = new_point
    return distance, sampled_points_a


def get_new_direction_vector(previous_vector, base_x, counter):
    return rotate_vector(previous_vector, counter + 1, base_x)


def calculate_skeleton_curvature(n, arc, number_of_points):
    t_list = np.linspace(0, 1, number_of_points).tolist()
    curvature = []
    for t in t_list:
        first_derivative = calculate_bezier_derivative(n, arc, t)
        second_derivative = calculate_bezier_second_derivative(n, arc, t)
        first_derivative = np.array(first_derivative)
        second_derivative = np.array(second_derivative)
        vector_product = np.cross(first_derivative, second_derivative)
        stevec = magnitude(vector_product)
        denominator = magnitude(first_derivative) ** 3
        curvature.append(stevec / denominator)
    return curvature


def perform_measurements(n, points, num_of_points, direction_vectors):
    distances = {}
    _, arc = perform_arc_length_parametrization_bezier_curve(n, points, num_of_points)
    if arc is not None:
        for i in range(1, len(arc) - 2):
            previous_point = arc[i - 1]
            current_point = arc[i]
            next_point = arc[i + 1]
            # normala
            a = next_point - current_point
            b = previous_point - current_point
            a = a / magnitude(a)
            b = b / magnitude(b)
            normal = np.cross(a, b)
            if np.any(np.isnan(normal)):
                return None, None, None
            normal = normal / magnitude(normal)
            base_y = np.cross(a, normal)
            base_y = base_y / magnitude(base_y)
            distances[i] = sample_rays(current_point, normal, object_points, a, base_y, arc, points)
        distance_start = uniform_sample_at_ends(arc[0], arc[1], 10000, object_points, direction_vectors, points, arc)
        distance_end = uniform_sample_at_ends(arc[len(arc) - 1], arc[len(arc) - 2], 10000, object_points, direction_vectors, points, arc)
        skeleton_curvature = calculate_skeleton_curvature(n, arc, num_of_points)
        # distances_skeleton_all[filename] = distances
        # distances_start_all[filename] = distance_start
        # distances_end_all[filename] = distance_end
    return distances, distance_start, distance_end, skeleton_curvature


if __name__ == '__main__':
    skeletons_folder = '../skeletons/'
    num_of_points = 10
    n = 5
    num_of_samples = 1000
    distances_skeleton_all, distances_start_all, distances_end_all, curvatures = {}, {}, {}, {}
    direction_vectors = sample_direction_vectors(num_of_samples)
    # plot_new_points(direction_vectors)
    for filename in os.listdir(skeletons_folder):
        print('processing', filename)
        # if filename != 'fib1-3-3-0_41.csv':
        #     continue

        points = read_file_collect_points(filename, skeletons_folder)
        object_points = read_nii_file('../extracted_data/', filename.replace('.csv', '.nii'))
        if points is None:
            print('no points for file', filename)
            continue
        distances, distance_start, distance_end, skeleton_curvature = perform_measurements(n, points, num_of_points, direction_vectors)
        if distances is None:
            new_n = n
            while distances is None:
                new_n -= 1
                print('try to form new distances with order', new_n)
                distances, distance_start, distance_end, skeleton_curvature = perform_measurements(new_n, points, num_of_points, direction_vectors)
        distances_skeleton_all[filename] = distances
        distances_start_all[filename] = distance_start
        distances_end_all[filename] = distance_end
        curvatures[filename] = skeleton_curvature

    # calculate_average(distances_skeleton_all, distances_start_all, distances_end_all)
    skeleton, start, end, curvature = group_distances(distances_skeleton_all, distances_start_all, distances_end_all, curvatures)
    sample_new_points(skeleton, start, end, curvature, 1)
