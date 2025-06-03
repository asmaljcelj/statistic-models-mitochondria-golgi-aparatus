import math
import os

import numpy as np

import bezier
import math_utils
import utils


def sample_direction_vectors(num_of_samples, skeleton_angle_increment=1):
    np.random.seed(10)
    # source: http://blog.thomaspoulet.fr/uniform-sampling-on-unit-hemisphere/
    u_samples = np.random.uniform(0, 1, num_of_samples)
    v_samples = np.random.uniform(0, 1, num_of_samples)
    u_samples = np.append(u_samples, [0], axis=0)
    v_samples = np.append(v_samples, [0], axis=0)
    # sample another
    result = math_utils.random_cosine(u_samples, v_samples, 1, skeleton_angle_increment)
    return result


def sample_at_ends(end_point, second_point, shape, points_on_hemisphere):
    normal = math_utils.normalize(end_point - second_point)
    R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal)
    sampled_points_t = {}
    distances = {}
    for direction_vector in points_on_hemisphere:
        rotated_vector = np.dot(R, direction_vector)
        distance, sampled_points = iterate_ray(end_point, rotated_vector, shape)
        key = utils.dict_key_from_point(direction_vector)
        distances[key] = distance
        sampled_points_t[key] = sampled_points
    return distances, sampled_points_t


def sample_rays(origin, direction_vector, shape, base_vector, angle_increment=5):
    original_direction_vector = direction_vector
    distances, sampled_points = [], {}
    angle = 0
    while angle < 360:
        angle += angle_increment
        distance, sampled_points_a = iterate_ray(origin, direction_vector, shape)
        distances.append([distance, angle])
        sampled_points[angle] = sampled_points_a
        direction_vector = math_utils.rotate_vector(original_direction_vector, angle, base_vector)
    return distances, sampled_points


def iterate_ray(origin, direction_vector, shape):
    current_point = origin
    sampled_points_a = [current_point]
    previous_voxel, current_voxel = None, [int(origin[0]), int(origin[1]), int(origin[2])]
    while True:
        new_point = current_point + direction_vector
        if math.isnan(new_point[0]) or math.isnan(new_point[1]) or math.isnan(new_point[2]):
            print()
        x_coord, y_coord, z_coord = int(new_point[0]), int(new_point[1]), int(new_point[2])
        previous_voxel = current_voxel
        current_voxel = [x_coord, y_coord, z_coord]
        in_boundary = x_coord >= 0 and y_coord >= 0 and z_coord >= 0 and x_coord < shape.scalle[0] and y_coord < shape.scalle[1] and z_coord < shape.scalle[2] and shape[x_coord][y_coord][z_coord] > 0
        if not in_boundary:
            boundary_voxel = [previous_voxel[0], previous_voxel[1], previous_voxel[2]]
            distance = math_utils.distance_between_points(origin, boundary_voxel)
            break
        sampled_points_a.append(new_point)
        current_point = new_point
    return distance, sampled_points_a


def calculate_skeleton_curvature_and_torsion(n, arc, number_of_points):
    t_list = np.linspace(0, 1, number_of_points + 1).tolist()
    curvature, torsions = [], []
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
        torsions.append(torsion)
        curvature.append(result)
    return curvature, torsions


def perform_measurements(n, skeleton_points, num_of_points, direction_vectors, object_points, angle_increment=1):
    # perform measurment
    skleton_distances = {}
    bezier_curve, arc, length = bezier.perform_arc_length_parametrization_bezier_curve(n, skeleton_points, num_of_points)
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
            skleton_distances[i], sampled_points = sample_rays(current_point, normal, vector_to_next_point, object_points, angle_increment)
        distances_start, sampled_start = sample_at_ends(arc[0], arc[1], object_points, direction_vectors)
        distances_end, sampled_end = sample_at_ends(arc[len(arc) - 1], arc[len(arc) - 2], object_points, direction_vectors)
        skeleton_curvature, skeleton_torsion = calculate_skeleton_curvature_and_torsion(n, bezier_curve, num_of_points)
    return skleton_distances, distances_start, distances_end, skeleton_curvature, length, skeleton_torsion


def sample(skeletons_folder, extracted_data_folder, testing_data, num_of_skeleton_points, n, direction_vectors, direction_with_angles, angle_increment):
    # samples each instance in extracted_data_folder with matching skeleton
    distances_skeleton_all, distances_start_all, distances_end_all, curvatures_all, lengths, torsions = {}, {}, {}, {}, [], {}
    for filename in os.listdir(skeletons_folder):
        print('processing', filename)
        skeleton_points = utils.read_file_collect_points(filename, skeletons_folder)
        object_points = utils.read_nii_file(extracted_data_folder, filename.replace('.csv', '.nii'))
        if skeleton_points is None:
            print('no skeleton for file', filename)
            continue
        distances, distance_start, distance_end, skeleton_curvature, length, torsion = perform_measurements(n,
                                                                                                            skeleton_points,
                                                                                                            num_of_skeleton_points,
                                                                                                            direction_vectors,
                                                                                                            object_points,
                                                                                                            angle_increment)
        if distances is None:
            new_n = n
            while distances is None:
                new_n -= 1
                print('try to form new distances with order', new_n)
                distances, distance_start, distance_end, skeleton_curvature, length, torsion = perform_measurements(
                    new_n, skeleton_points, num_of_skeleton_points, direction_vectors, object_points, angle_increment)
        distances_skeleton_all[filename] = distances
        distances_start_all[filename] = distance_start
        distances_end_all[filename] = distance_end
        curvatures_all[filename] = skeleton_curvature
        torsions[filename] = torsion
        lengths.append(length)
    if not testing_data:
        skeleton, start, end, curvature, torsion = utils.group_distances(distances_skeleton_all, distances_start_all,
                                                                     distances_end_all, curvatures_all, torsions)
        utils.save_measurements_to_file('measurements/learn/measurements.pkl', skeleton, start, end, curvature, lengths,
                                    direction_with_angles, torsion)
    else:
        for filename in distances_skeleton_all:
            distances_skeleton = distances_skeleton_all[filename]
            distances_start = distances_start_all[filename]
            distances_end = distances_end_all[filename]
            curvatures = curvatures_all[filename]
            torsion = torsions[filename]
            name = 'measurements_' + filename
            name = name.replace('.csv', '.nii')
            utils.save_measurements_to_file('measurements/testing/' + name, distances_skeleton, distances_start, distances_end, curvatures,
                                            None,
                                            direction_with_angles, torsion)


skeletons_folder = 'skeletons/learn'
num_of_skeleton_points, n, num_of_samples = 15, 5, 1000
angle_increment = 3
if 360 % angle_increment != 0:
    raise Exception('angle increment has to be a multiple of 360.')
distances_skeleton_all, distances_start_all, distances_end_all, curvatures_all, lengths, torsions = {}, {}, {}, {}, [], {}
direction_vectors, direction_with_angles = sample_direction_vectors(num_of_samples, angle_increment)
print('starting sampling learning group')
sample('skeletons/learn/', 'extracted_data/learning/', False, num_of_skeleton_points, n, direction_vectors, direction_with_angles, angle_increment)
print('starting sampling testing group')
sample('skeletons/test/', 'extracted_data/test/', True, num_of_skeleton_points, n, direction_vectors, direction_with_angles, angle_increment)
