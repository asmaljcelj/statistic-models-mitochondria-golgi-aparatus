import math
import os

import numpy as np
import trimesh

import bezier
import ga_extract
import ga_statistics
import math_utils
import mitochondria_sampling
import mitochondria_skeletonization
import utils


def calculate_rmse_for_mito(new_object_path, testing_directory):
    direction_vectors, direction_with_angles = mitochondria_sampling.sample_direction_vectors(1000, 3)
    mesh = trimesh.load(new_object_path)
    voxelized = mesh.voxelized(pitch=1.0)
    filled = voxelized.fill()

    # Get list of voxel coordinates (as numpy array)
    voxels = filled.points.astype(int)
    image = utils.create_3d_image(voxels)
    result = mitochondria_skeletonization.skeletonize_voxels(np.array(image))
    bezier_curve, arc, length = bezier.perform_arc_length_parametrization_bezier_curve(5, result, 15)
    skeleton_distances = {}
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
        if math.isnan(normal[0]) or math.isnan(normal[1]) or math.isnan(normal[2]):
            normal = np.array([0, 1, 0])
        skeleton_distances[i], _ = mitochondria_sampling.sample_rays(current_point, normal, vector_to_next_point, image, 3)
    distances_start, _ = mitochondria_sampling.sample_at_ends(arc[0], arc[1], image, direction_vectors)
    distances_end, _ = mitochondria_sampling.sample_at_ends(arc[len(arc) - 1], arc[len(arc) - 2], image, direction_vectors)
    total_rmse, num_of_testing_intances = 0, 0
    for filename in os.listdir(testing_directory):
        print('processing', filename)
        total_rmse += calculate_rmse_between_objects(distances_start, distances_end, skeleton_distances, filename)
        num_of_testing_intances += 1
    return round(total_rmse / num_of_testing_intances, 3)


def calculate_rmse_for_golgi(new_object_path, testing_directory):
    direction_vectors = math_utils.generate_direction_vectors(
        np.array([[1, 0],
                  [0, 1]]), 10)
    mesh = trimesh.load(new_object_path)
    voxelized = mesh.voxelized(pitch=1.0)
    filled = voxelized.fill()
    # Get list of voxel coordinates (as numpy array)
    voxels = filled.points.astype(int)
    cisternae, eigenvectors = ga_extract.read_files(np.array(voxels))
    distances = {}
    for i, cis in enumerate(cisternae):
        centers = utils.cisterna_volume_extraction(cis)
        measurements = ga_statistics.calculate_distances_to_landmark_points(centers, direction_vectors)
        distances[i] = measurements
    total_rmse, num_of_testing_instances = 0, 0
    for filename in os.listdir(testing_directory):
        print('processing', filename)
        rmse_one_instance = calculate_rmse_between_ga_objects(distances, filename, len(direction_vectors))
        total_rmse += rmse_one_instance
        print('rmse:', rmse_one_instance)
        num_of_testing_instances += 1
    return round(total_rmse / num_of_testing_instances, 3)


def calculate_rmse_between_objects(new_start, new_end, new_skeleton, test_object):
    _, start, end, skeleton, _, _, _ = utils.read_measurements_from_file('measurements/testing/' + test_object)
    rmse, num_instances = 0, 0
    rmse_value, n = rmse_calculate_edges(new_start, start)
    num_instances += n
    rmse += rmse_value
    rmse_value, n = rmse_calculate_edges(new_end, end)
    num_instances += n
    rmse += rmse_value
    rmse_value, n = rmse_calculate_skeleton(new_skeleton, skeleton)
    num_instances += n
    rmse += rmse_value
    return math.sqrt(rmse / num_instances)


def calculate_rmse_between_ga_objects(distances, test_object, direction_vectors):
    data = utils.read_measurements_from_file_ga('measurements_ga/testing/' + test_object)
    return rmse_calculate_ga(distances, data, direction_vectors)


def rmse_calculate_ga(actual, testing, direction_vectors):
    rmse = 0
    smaller_instance, larger_instance = (actual, testing) if len(actual) < len(testing) else (testing, actual)
    interval = np.linspace(0, len(larger_instance) - 1, len(smaller_instance)).astype(int)
    calculations = 0
    for i, data in enumerate(smaller_instance):
        if isinstance(smaller_instance, dict):
            measurements = smaller_instance[data]
        else:
            measurements = data
        if len(measurements) > 1:
            measurements = [list(np.mean(measurements, axis=0))]
        larger_index = interval[i]
        for j, measurement in enumerate(measurements):
            if len(larger_instance[larger_index]) == 0:
                continue
            other_value = larger_instance[larger_index][j]
            for k, m in enumerate(measurement):
                calculations += 1
                rmse += (m - other_value[k]) ** 2
    return rmse / (len(smaller_instance) * direction_vectors)


def rmse_calculate_edges(actual, testing):
    rmse = 0
    for key in actual:
        actual_value = actual[key]
        test_value = testing[key]
        rmse += (actual_value - test_value) ** 2
    return rmse, len(actual)


def rmse_calculate_skeleton(actual, testing):
    rmse, num_instances = 0, 0
    for key in actual:
        actual_values = actual[key]
        test_values = testing[key]
        for i, actual_value_and_angle in enumerate(actual_values):
            actual_value = actual_value_and_angle[0]
            test_value = test_values[i][0]
            rmse += (actual_value - test_value) ** 2
        num_instances += len(actual_values)
    return rmse, num_instances


# mito
mito_filename = 'results/smooth_025_10_123.obj'
rmse_mito = calculate_rmse_for_mito(mito_filename, 'measurements/testing/')
print('rmse for mitochondria instance', mito_filename, ':', rmse_mito)
# GA
ga_filename = 'results/90_50_i_1_s_02_b_20.obj'
rmse_ga = calculate_rmse_for_golgi(ga_filename, 'measurements_ga/testing/')
print('rmse for Golgi apparatus instance', ga_filename, ':', rmse_ga)
