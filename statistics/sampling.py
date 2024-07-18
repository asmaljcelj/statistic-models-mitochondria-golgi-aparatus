import os

import numpy as np

import bezier
import math_utils
import outside_statistics
import utils


def sample_direction_vectors(num_of_samples):
    # source: http://blog.thomaspoulet.fr/uniform-sampling-on-unit-hemisphere/
    u_samples = np.random.uniform(0, 1, num_of_samples)
    v_samples = np.random.uniform(0, 1, num_of_samples)
    u_samples = np.append(u_samples, [0], axis=0)
    v_samples = np.append(v_samples, [0], axis=0)
    # sample another
    return math_utils.random_cosine(u_samples, v_samples, 1)


def sample_at_ends(end_point, second_point, shape, points_on_hemisphere):
    normal = math_utils.normalize(end_point - second_point)
    R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal)
    sampled_points_t = {}
    distances = {}
    for direction_vector in points_on_hemisphere:
        rotated_vector = np.dot(R, direction_vector)
        distance, sampled_points = iterate_ray(end_point, rotated_vector, shape)
        key = (direction_vector[0], direction_vector[1], direction_vector[2])
        distances[key] = distance
        sampled_points_t[key] = sampled_points
    return distances


# todo: neki cudno sampla tudi izven BB-ja???????? (treba popravit)
def sample_rays(origin, direction_vector, shape, base_vector, angle_increment=1):
    original_direction_vector = direction_vector
    distances, sampled_points = [], {}
    angle = 0
    while angle < 360:
        angle += angle_increment
        distance, sampled_points_a = iterate_ray(origin, direction_vector, shape)
        distances.append([distance, angle])
        sampled_points[angle] = sampled_points_a
        direction_vector = math_utils.rotate_vector(original_direction_vector, angle, base_vector)
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
            distance = math_utils.distance_between_points(origin, boundary_voxel)
            break
        sampled_points_a.append(new_point)
        current_point = new_point
    return distance, sampled_points_a


def calculate_skeleton_curvature_and_torsion(n, arc, number_of_points):
    t_list = np.linspace(0, 1, number_of_points).tolist()
    curvature = []
    for t in t_list:
        first_derivative = np.array(bezier.calculate_bezier_derivative(n, arc, t))
        second_derivative = np.array(bezier.calculate_bezier_second_derivative(n, arc, t))
        third_derivative = np.array(bezier.calculate_bezier_third_derivative(n, arc, t))
        vector_product = np.cross(first_derivative, second_derivative)
        stevec = math_utils.magnitude(vector_product)
        denominator = math_utils.magnitude(first_derivative) ** 3
        torsion = np.dot(vector_product, third_derivative) / (stevec ** 2)
        result = stevec / denominator
        if torsion < 0:
            result *= -1
        curvature.append(result)
    return curvature


def perform_measurements(n, points, num_of_points, direction_vectors, object_points):
    skleton_distances = {}
    _, arc = bezier.perform_arc_length_parametrization_bezier_curve(n, points, num_of_points)
    # todo: v model vkljuci tudi dolzino skeletona
    if arc is not None:
        for i in range(1, len(arc) - 2):
            previous_point = arc[i - 1]
            current_point = arc[i]
            next_point = arc[i + 1]
            vector_to_next_point = math_utils.normalize(next_point - current_point)
            vector_to_previous_point = math_utils.normalize(previous_point - current_point)
            normal = np.cross(vector_to_next_point, vector_to_previous_point)
            if np.any(np.isnan(normal)):
                return None, None, None
            normal = math_utils.normalize(normal)
            skleton_distances[i] = sample_rays(current_point, normal, object_points, vector_to_next_point)
        distances_start = sample_at_ends(arc[0], arc[1], object_points, direction_vectors)
        distances_end = sample_at_ends(arc[len(arc) - 1], arc[len(arc) - 2], object_points, direction_vectors)
        skeleton_curvature = calculate_skeleton_curvature_and_torsion(n, arc, num_of_points)
    return skleton_distances, distances_start, distances_end, skeleton_curvature


if __name__ == '__main__':
    skeletons_folder = '../skeletons/'
    num_of_points, n, num_of_samples, num_files = 10, 5, 500, 1
    distances_skeleton_all, distances_start_all, distances_end_all, curvatures_all = {}, {}, {}, {}
    direction_vectors, direction_with_angles = sample_direction_vectors(num_of_samples)
    for filename in os.listdir(skeletons_folder):
        print('processing', filename)
        points = utils.read_file_collect_points(filename, skeletons_folder)
        object_points = utils.read_nii_file('../extracted_data/', filename.replace('.csv', '.nii'))
        if points is None:
            print('no points for file', filename)
            continue
        distances, distance_start, distance_end, skeleton_curvature = perform_measurements(n, points, num_of_points, direction_vectors, object_points)
        if distances is None:
            new_n = n
            while distances is None:
                new_n -= 1
                print('try to form new distances with order', new_n)
                distances, distance_start, distance_end, skeleton_curvature = perform_measurements(new_n, points, num_of_points, direction_vectors, object_points)
        distances_skeleton_all[filename] = distances
        distances_start_all[filename] = distance_start
        distances_end_all[filename] = distance_end
        curvatures_all[filename] = skeleton_curvature

    skeleton, start, end, curvature = utils.group_distances(distances_skeleton_all, distances_start_all, distances_end_all, curvatures_all)
    outside_statistics.sample_new_points(skeleton, start, end, curvature, num_files, direction_with_angles)
